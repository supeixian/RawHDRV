from __future__ import division
import os
import argparse
from config import get_train_config

# 初始化配置
parser = argparse.ArgumentParser(description='HDR Video Training')
parser.add_argument('--model', type=str, default='RRVSR_HDR', help='模型名称')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--scale', type=int, default=1, help='HDR固定缩放比例')
parser.add_argument('--continue_train', action='store_true', help='是否从检查点恢复训练')
args = parser.parse_args()
opt = get_train_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import torch.nn as nn
import torch.optim as optim
# from pytorch_msssim import ssim
import math
# import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from models.model import RRVSR
from models.RawHDRV import RawHDRV
from data.dataset import HDRVideoDataset
from tqdm import tqdm
from data.process import process
from tensorboardX import SummaryWriter
from utils import get_psnr, get_ssim
import cv2
import numpy as np

# 检查点配置
checkpoint_dir = os.path.join(opt.weight_savepath, opt.model)
os.makedirs(os.path.join(checkpoint_dir, 'checkpoints'), exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练状态初始化
best_psnr = 0.0
best_loss = float('inf')
start_epoch = 0
# color_correction = color_correction()

# 模型初始化
# model = RRVSR(nf=64, nframes=opt.N_frames, scale=opt.scale).to(device)
model = RawHDRV(num_feat=64,
                    RB_gudie=True, 
                    G_guidance=True,
                    mask_guide=True,
                    num_blocks=[2,3,4,1],  # [level1, level2, level3, latent, refinement]
                    spynet_path=None,
                    heads=[1,2,4,8],         # [level1, level2, level3, latent] - 增加了level3的8个头
                    ffn_expansion_factor=2.66,
                    softmask=False,
                    softblending=False,
                    bias=False,
                    LayerNorm_type='BiasFree',
                    ch_compress=True,
                    squeeze_factor=[4, 4, 4],
                    masked=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)  # 每50个epoch降低10倍

# 加载检查点
if args.continue_train:
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints/epoch_86.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        # ✅ 重新创建scheduler，指定正确的last_epoch
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, 
                                              last_epoch=checkpoint['epoch'])
        best_psnr = checkpoint['best_psnr']
        current_lr = optimizer.param_groups[0]['lr']
        print(f'从epoch {checkpoint["epoch"]}恢复训练，PSNR: {checkpoint["psnr"]:.2f} dB, 当前LR: {current_lr:.6f}')

# HDR损失函数
class l1_loss():
    def __init__(self, weight=1.0):
        self.weight = weight
        # 用于在外部读取每个通道的 loss
        self.per_channel = None  

    def __call__(self, pred, target):
        """
        pred, target: [B, C, H, W]
        返回:
          total_loss: 标量，可以直接 .backward()
          per_channel: [C] 各通道的 L1 loss（可用于日志或加权）
        """
        # 1) 逐元素绝对误差，不做 reduce
        diff = F.l1_loss(pred, target, reduction='none')  # [B, C, H, W]
        # 2) 按 (B, H, W) 三个维度求平均，得到每个通道的 loss
        #    结果 self.per_channel 形状 [C]
        self.per_channel = diff.mean(dim=(0, 2, 3))        

        # 3) 标量 loss：对通道求平均再乘权重
        total_loss = self.per_channel.mean() * self.weight

        return total_loss
    
class log_l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(torch.log(torch.clamp(pred, min=0)+1/5000), torch.log(target+1/5000))
    

class SimpleOverexposureColorLoss(nn.Module):
    """混合颜色损失：全局损失 + 过曝区域加权损失"""
    
    def __init__(self, weight=0.3, smooth_weight=0.1, overexposure_weight=2.0):
        super().__init__()
        self.weight = weight
        self.smooth_weight = smooth_weight
        self.overexposure_weight = overexposure_weight
        
    def forward(self, pred_hdr, target_hdr, mask_over=None):
        # 过曝区域加权损失（如果提供mask）
        overexposure_loss = 0
        if mask_over is not None:
            if mask_over.shape[1] == 1:
                mask_over = mask_over.repeat(1, pred_hdr.shape[1], 1, 1)
            
            masked_diff = torch.abs(pred_hdr - target_hdr) * mask_over
            mask_sum = torch.sum(mask_over, dim=[2, 3], keepdim=True) + 1e-8
            overexposure_loss = torch.mean(torch.sum(masked_diff, dim=[2, 3], keepdim=True) / mask_sum)
        
        # 3. 全局色彩平滑约束
        grad_x = torch.abs(pred_hdr[:, :, :, 1:] - pred_hdr[:, :, :, :-1])
        grad_y = torch.abs(pred_hdr[:, :, 1:, :] - pred_hdr[:, :, :-1, :])
        
        smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)
        
        # 组合损失
        total_loss =  self.overexposure_weight * overexposure_loss + self.smooth_weight * smooth_loss
        
        return self.weight * total_loss


class L1_ulaw(nn.Module):
    """
    L1 loss on μ-law tonemapped HDR images (mean reduction).
    T(x) = log(1 + mu * x) / log(1 + mu)
    Loss = mean(|T(pred) - T(gt)|)

    Usage:
        loss_fn = L1_ulaw(mu=5000.0)
        loss = loss_fn(pred, gt)  # scalar tensor
    """
    def __init__(self, mu: float = 5000.0):
        super().__init__()
        self.mu = float(mu)
        # precompute denominator scalar
        self._den = float(math.log1p(self.mu))

    def _tonemap(self, x: torch.Tensor) -> torch.Tensor:
        # ensure float dtype and non-negative values
        x = x.float().clamp(min=0.0)
        return torch.log1p(self.mu * x) / self._den

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: same shape tensors (e.g., [B,C,H,W]).
        Returns: scalar tensor (mean L1 between tonemapped pred and gt).
        """
        t_pred = self._tonemap(pred)
        t_gt   = self._tonemap(gt)
        return torch.mean(torch.abs(t_pred - t_gt))


loss_logl2 = log_l2_loss(weight=1)
loss_l1 = l1_loss(weight=1)
loss_l1ulaw = L1_ulaw()
simple_overexp_loss = SimpleOverexposureColorLoss(weight=0.3).to(device)
loss_mask = l1_loss(weight=0.5)

# 数据加载
train_dataset = HDRVideoDataset(opt, mode='train')
train_loader = DataLoader(train_dataset, 
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        prefetch_factor=1,
                        persistent_workers=True)

valid_dataset = HDRVideoDataset(opt, mode='test')
valid_loader = DataLoader(valid_dataset,
                        batch_size=opt.valid_batch_size,  
                        shuffle=False, 
                        num_workers=2,
                        pin_memory=True,
                        prefetch_factor=1,
                        persistent_workers=True) 

print(f"训练集样本总数: {len(train_dataset)}")
print(f"验证集样本总数: {len(valid_dataset)}")
# 日志记录
writer = SummaryWriter(f'./logs/hdr_train/{opt.model}_scale{opt.scale}')

def augment_rgbg_data_torch(ldr_data, hdr_data):
    """
    使用PyTorch原生操作进行图像增强
    Args:
        ldr_data: torch.Tensor, shape [B, T, 4, H, W] 或 [B, 4, H, W]
        hdr_data: torch.Tensor, shape [B, 4, H, W]
    """
    # 随机水平翻转
    if torch.rand(1).item() < 0.5:
        ldr_data = torch.flip(ldr_data, dims=[-1])
        hdr_data = torch.flip(hdr_data, dims=[-1])
    
    # 随机垂直翻转
    if torch.rand(1).item() < 0.5:
        ldr_data = torch.flip(ldr_data, dims=[-2])
        hdr_data = torch.flip(hdr_data, dims=[-2])
    
    # 随机旋转90度的倍数
    rotation = torch.randint(0, 4, (1,)).item()
    if rotation > 0:
        # 对于90度和270度旋转，需要交换H和W维度
        if rotation == 1:  # 90度
            ldr_data = torch.rot90(ldr_data, k=1, dims=[-2, -1])
            hdr_data = torch.rot90(hdr_data, k=1, dims=[-2, -1])
        elif rotation == 2:  # 180度
            ldr_data = torch.rot90(ldr_data, k=2, dims=[-2, -1])
            hdr_data = torch.rot90(hdr_data, k=2, dims=[-2, -1])
        elif rotation == 3:  # 270度
            ldr_data = torch.rot90(ldr_data, k=3, dims=[-2, -1])
            hdr_data = torch.rot90(hdr_data, k=3, dims=[-2, -1])
    
    return ldr_data, hdr_data

def train(epoch):
    scaler = torch.cuda.amp.GradScaler()  # 新增
    global best_psnr, best_loss
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, samples in enumerate(progress_bar):
        # 数据加载
        LRs_RAW = samples['LDRs_RAW'].to(device, non_blocking=True)
        HR_HDR = samples['HDR_DATA'].to(device, non_blocking=True)
        wb = samples['wb'].to(device, non_blocking=True)
        cam2rgb = samples['cam2rgb'].to(device, non_blocking=True)
        
        # 应用数据增强（直接在GPU上操作）
        # if torch.rand(1).item() < 0.5:  # 50%概率应用增强
        #     LRs_RAW, HR_HDR = augment_rgbg_data_torch(LRs_RAW, HR_HDR)
        # LR_RGB = samples['LDR_DATA'].to(device, non_blocking=True)
        # print(f"HR_HDR shape: {HR_HDR.shape}")

        # 前向传播
        pred_HDR, under_mask, over_mask = model(LRs_RAW, wb, cam2rgb)
        LRs_RAW = LRs_RAW[:, LRs_RAW.shape[1] // 2]  # 变为 (B, 1, C, H, W)
        
        log_loss = loss_logl2(pred_HDR, HR_HDR)
        l1_loss = loss_l1(pred_HDR, HR_HDR)
        overexp_color_loss = simple_overexp_loss(pred_HDR, HR_HDR, over_mask)
        loss = log_loss + l1_loss + overexp_color_loss


        total_loss += loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 修改反向传播
        scaler.unscale_(optimizer)  # ✅ 在clip前unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        
        # 当前迭代数（用于tensorboard记录）
        current_iter = epoch * len(train_loader) + batch_idx + 1
        
        # 日志记录
        writer.add_scalar('train/loss', loss.item(), current_iter)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 每个epoch结束时进行验证和保存
    avg_loss = total_loss / len(train_loader)
    
    # 执行验证
    val_psnr, val_ssim, val_loss = validate(epoch, current_iter)
    
    # 保存epoch检查点
    epoch_checkpoint = {
        'epoch': epoch,
        'iter': current_iter,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'psnr': val_psnr,
        'ssim': val_ssim,
        'loss': val_loss,
        'avg_train_loss': avg_loss
    }
    
    # 保存当前epoch的检查点
    if epoch % 2 == 0:
        torch.save(epoch_checkpoint, os.path.join(checkpoint_dir, f'checkpoints/epoch_{epoch}.pth'))
    

    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印epoch结果
    print(f"Epoch {epoch}: Val PSNR-L:{val_psnr:.2f} SSIM-L:{val_ssim:.4f} Avg_Loss:{avg_loss:.4f} | LR: {current_lr:.6f}")
    # print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 更新最佳模型
    if val_psnr >= best_psnr:
        best_psnr = val_psnr
        torch.save(epoch_checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
    print(f"最佳模型！PSNR-L: {best_psnr:.2f} dB\n")
    
    return avg_loss

def preprocess_for_log_metrics(gt, pred, eps=1e-6, normalize=True):
    """
    用于计算 PSNR-L / SSIM-L 的简单预处理：
      1. clamp 到 >=0
      2. log(1 + x)
      3. 归一化到 [0,1]（可选）

    参数:
      gt, pred: torch.Tensor, 形状一致 [B,C,H,W] 或 [C,H,W]
    返回:
      gt_proc, pred_proc: torch.Tensor, 处理后的张量
    """
    gt_log  = torch.log1p(torch.clamp(gt, min=0.0) + eps)
    pred_log = torch.log1p(torch.clamp(pred, min=0.0) + eps)

    if normalize:
        joint_max = torch.max(torch.cat([gt_log.flatten(), pred_log.flatten()]))
        gt_log  = gt_log / (joint_max + eps)
        pred_log = pred_log / (joint_max + eps)

    return gt_log, pred_log

def validate(epoch, current_iter):
    model.eval()
    psnr_values = []
    ssim_values = []
    total_loss = 0.0
    
    with torch.no_grad():
        # for samples in tqdm(valid_loader, desc='验证中'):
        for idx, samples in enumerate(tqdm(valid_loader, desc='验证中')):
            LRs_RAW = samples['LDRs_RAW'].to(device, non_blocking=True)
            HR_HDR = samples['HDR_DATA'].to(device, non_blocking=True)
            wb = samples['wb'].to(device, non_blocking=True)
            cam2rgb = samples['cam2rgb'].to(device, non_blocking=True)

            pred_HDR, under_mask, over_mask = model(LRs_RAW, wb, cam2rgb)

            LRs_RAW = LRs_RAW[:, LRs_RAW.shape[1] // 2].unsqueeze(1)  # 变为 (B, 1, C, H, W)
            if HR_HDR.dim() == 5:
                # 合并batch和sequence维度 [B,T,C,H,W] -> [B*T,C,H,W]
                HR_HDR = HR_HDR.view(-1, HR_HDR.size(2), HR_HDR.size(3), HR_HDR.size(4))
            if LRs_RAW.dim() == 5:
                # 合并batch和sequence维度 [B,T,C,H,W] -> [B*T,C,H,W]
                LRs_RAW = LRs_RAW.view(-1, LRs_RAW.size(2), LRs_RAW.size(3), LRs_RAW.size(4))

            log_loss = loss_logl2(pred_HDR, HR_HDR)
            l1_loss = loss_l1(pred_HDR, HR_HDR)
            overexp_color_loss = simple_overexp_loss(pred_HDR, HR_HDR, over_mask)
            loss = log_loss + l1_loss + overexp_color_loss

            total_loss += loss.item()

            # 添加可视化保存 ↓↓↓
            if idx % 20 == 0:  # 每1个样本保存一次
                save_path = f'./results_RVours_mask/epoch{epoch}/'  # 修改：移除iter参数
                os.makedirs(save_path, exist_ok=True)

                # 保存掩码可视化
                mask_u_vis = (under_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
                mask_o_vis = (over_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
                
                cv2.imwrite(f'{save_path}sample{idx}_mask_u_pred.png', mask_u_vis)
                cv2.imwrite(f'{save_path}sample{idx}_mask_o_pred.png', mask_o_vis)

                out_hdr = process(pred_HDR.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
                gt_hdr = process(HR_HDR.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
                in_hdr = process(LRs_RAW.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=1.0)[0].numpy().transpose((1,2,0))
            
                out_hdr = (out_hdr*255).astype(np.uint8)
                gt_hdr = (gt_hdr*255).astype(np.uint8)
                in_hdr = (in_hdr*255).astype(np.uint8)  

                cv2.imwrite(f'{save_path}sample{idx}_input.png', in_hdr)
                cv2.imwrite(f'{save_path}sample{idx}_pred.png', out_hdr)
                cv2.imwrite(f'{save_path}sample{idx}_gt.png', gt_hdr)

            
            total_loss += loss.item()

            gt_log, pred_log = preprocess_for_log_metrics(HR_HDR, pred_HDR)

            psnr = get_psnr(gt_log, pred_log)
            ssim = get_ssim(gt_log, pred_log)  # 新增SSIM计算
            psnr_values.append(psnr)
            ssim_values.append(ssim)  # 收集SSIM值
            
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)  # 计算平均SSIM
    avg_loss = total_loss / len(valid_loader)

    # 记录验证指标
    writer.add_scalar('valid/loss', avg_loss, current_iter)
    writer.add_scalar('valid/ssim', avg_ssim, current_iter)
    writer.add_scalar('valid/psnr', avg_psnr, current_iter)
    
    model.train()  # 重新设置为训练模式
    return avg_psnr, avg_ssim, avg_loss


writer.close()

# 在文件底部修改主入口
if __name__ == '__main__':
    # 确保所有初始化操作在main中完成
    for epoch in range(start_epoch, opt.number_epochs + 1):
        train(epoch)
        scheduler.step()

    writer.close()