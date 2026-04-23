from __future__ import division
import os
import argparse
from config import get_train_config

parser = argparse.ArgumentParser(description='HDR Video Training')
parser.add_argument('--model', type=str, default='RawHDRV', help='Model name')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--continue_train', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()
opt = get_train_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.RawHDRV import RawHDRV
from data.dataset import HDRVideoDataset
from tqdm import tqdm
from data.process import process
from tensorboardX import SummaryWriter
from utils import get_psnr, get_ssim, l1_loss, log_l2_loss, SimpleOverexposureColorLoss, L1_ulaw
import cv2
import numpy as np

checkpoint_dir = os.path.join(opt.weight_savepath, opt.model)
os.makedirs(os.path.join(checkpoint_dir, 'checkpoints'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_psnr = 0.0
best_loss = float('inf')
start_epoch = 0

model = RawHDRV(num_feat=64,
                    RB_gudie=True, 
                    G_guidance=True,
                    mask_guide=True,
                    num_blocks=[2,3,4,1],
                    spynet_path=None,
                    heads=[1,2,4,8],
                    ffn_expansion_factor=2.66,
                    softmask=False,
                    softblending=False,
                    bias=False,
                    LayerNorm_type='BiasFree',
                    ch_compress=True,
                    squeeze_factor=[4, 4, 4],
                    masked=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

if args.continue_train:
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints/epoch_86.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, 
                                              last_epoch=checkpoint['epoch'])
        best_psnr = checkpoint['best_psnr']
        current_lr = optimizer.param_groups[0]['lr']
        print(f'从epoch {checkpoint["epoch"]}恢复训练，PSNR: {checkpoint["psnr"]:.2f} dB, 当前LR: {current_lr:.6f}')

loss_logl2 = log_l2_loss(weight=1)
loss_l1 = l1_loss(weight=1)
loss_l1ulaw = L1_ulaw()
simple_overexp_loss = SimpleOverexposureColorLoss(weight=0.3).to(device)
loss_mask = l1_loss(weight=0.5)

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
writer = SummaryWriter(f'./logs/hdr_train/{opt.model}')

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
    scaler = torch.cuda.amp.GradScaler()
    global best_psnr, best_loss
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, samples in enumerate(progress_bar):
        LRs_RAW = samples['LDRs_RAW'].to(device, non_blocking=True)
        HR_HDR = samples['HDR_DATA'].to(device, non_blocking=True)
        wb = samples['wb'].to(device, non_blocking=True)
        cam2rgb = samples['cam2rgb'].to(device, non_blocking=True)

        # Apply data augmentation
        # if torch.rand(1).item() < 0.5:
        #     LRs_RAW, HR_HDR = augment_rgbg_data_torch(LRs_RAW, HR_HDR)
        # LR_RGB = samples['LDR_DATA'].to(device, non_blocking=True)
        # print(f"HR_HDR shape: {HR_HDR.shape}")

        pred_HDR, under_mask, over_mask = model(LRs_RAW, wb, cam2rgb)
        LRs_RAW = LRs_RAW[:, LRs_RAW.shape[1] // 2]
        
        log_loss = loss_logl2(pred_HDR, HR_HDR)
        l1_loss = loss_l1(pred_HDR, HR_HDR)
        # l1_lossulaw = loss_l1ulaw(pred_HDR, HR_HDR)
        overexp_color_loss = simple_overexp_loss(pred_HDR, HR_HDR, over_mask)
        # loss = l1_loss
        # loss = l1_lossulaw
        # loss = log_loss + l1_loss
        loss = log_loss + l1_loss + overexp_color_loss

        total_loss += loss.item()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        
        current_iter = epoch * len(train_loader) + batch_idx + 1
        
        writer.add_scalar('train/loss', loss.item(), current_iter)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)

    val_psnr, val_ssim, val_loss = validate(epoch, current_iter)

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

    if epoch % 2 == 0:
        torch.save(epoch_checkpoint, os.path.join(checkpoint_dir, f'checkpoints/epoch_{epoch}.pth'))
    

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch}: Val PSNR-L:{val_psnr:.2f} SSIM-L:{val_ssim:.4f} Avg_Loss:{avg_loss:.4f} | LR: {current_lr:.6f}")

    if val_psnr >= best_psnr:
        best_psnr = val_psnr
        torch.save(epoch_checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
    print(f"最佳模型！PSNR-L: {best_psnr:.2f} dB\n")
    
    return avg_loss

def preprocess_for_log_metrics(gt, pred, eps=1e-6, normalize=True):
    """Simple preprocessing for PSNR-L / SSIM-L metrics."""
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
        for idx, samples in enumerate(tqdm(valid_loader, desc='验证中')):
            LRs_RAW = samples['LDRs_RAW'].to(device, non_blocking=True)
            HR_HDR = samples['HDR_DATA'].to(device, non_blocking=True)
            wb = samples['wb'].to(device, non_blocking=True)
            cam2rgb = samples['cam2rgb'].to(device, non_blocking=True)

            pred_HDR, under_mask, over_mask = model(LRs_RAW, wb, cam2rgb)

            LRs_RAW = LRs_RAW[:, LRs_RAW.shape[1] // 2].unsqueeze(1)
            if HR_HDR.dim() == 5:
                HR_HDR = HR_HDR.view(-1, HR_HDR.size(2), HR_HDR.size(3), HR_HDR.size(4))
            if LRs_RAW.dim() == 5:
                LRs_RAW = LRs_RAW.view(-1, LRs_RAW.size(2), LRs_RAW.size(3), LRs_RAW.size(4))

            log_loss = loss_logl2(pred_HDR, HR_HDR)
            l1_loss = loss_l1(pred_HDR, HR_HDR)
            # l1_lossulaw = loss_l1ulaw(pred_HDR, HR_HDR)
            overexp_color_loss = simple_overexp_loss(pred_HDR, HR_HDR, over_mask)
            # loss = l1_loss
            # loss = l1_lossulaw
            # loss = log_loss + l1_loss
            loss = log_loss + l1_loss + overexp_color_loss

            total_loss += loss.item()

            if idx % 20 == 0:
                save_path = f'./results_RVours_mask/epoch{epoch}/'
                os.makedirs(save_path, exist_ok=True)

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

            gt_log, pred_log = preprocess_for_log_metrics(HR_HDR, pred_HDR)

            psnr = get_psnr(gt_log, pred_log)
            ssim = get_ssim(gt_log, pred_log)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_loss = total_loss / len(valid_loader)

    writer.add_scalar('valid/loss', avg_loss, current_iter)
    writer.add_scalar('valid/ssim', avg_ssim, current_iter)
    writer.add_scalar('valid/psnr', avg_psnr, current_iter)
    
    model.train()
    return avg_psnr, avg_ssim, avg_loss

if __name__ == '__main__':
    for epoch in range(start_epoch, opt.number_epochs + 1):
        train(epoch)
        scheduler.step()

    writer.close()