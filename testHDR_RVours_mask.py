from __future__ import division
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 必须在导入 cv2 之前设置
import argparse
import cv2
from config import get_test_config

parser = argparse.ArgumentParser(description='HDR Video Testing')
parser.add_argument('--model', type=str, default='RRVSR_HDR', help='模型名称')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--scale', type=int, default=1, help='HDR固定缩放比例')
parser.add_argument('--save_image', type=bool, default=False, help='是否保存HDR结果')
parser.add_argument('--save_flows', action='store_true', help='是否保存光流可视化')
# parser.add_argument('--crop_coords', type=int, nargs=2, default=[800, 600], 
#                    help='固定裁剪坐标 (x, y)')
# parser.add_argument('--crop_size_custom', type=int, nargs=2, default=[512,512],
#                    help='固定裁剪尺寸 (width, height)')
args = parser.parse_args()
opt = get_test_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.RawHDRV import RawHDRV, Attention, CrossAttention
import utils
from data.dataset import HDRVideoDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pytorch_msssim import ms_ssim
from data.process import process, process_exr
import copy
import time
from thop import profile, clever_format

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载HDR专用模型
net = RawHDRV(num_feat=64,
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


def count_parameters(model, trainable_only=False):
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


def count_attention_ops(module, inputs, output):
    """Approximate MACs for channel self-attention matmuls only."""
    x = inputs[0]
    if x.dim() != 4:
        return

    b, c, h, w = x.shape
    num_heads = module.num_heads
    c_per_head = c // num_heads
    hw = h * w

    qk_macs = b * num_heads * c_per_head * c_per_head * hw
    av_macs = b * num_heads * c_per_head * hw * c_per_head
    module.total_ops += torch.DoubleTensor([int(qk_macs + av_macs)])


def count_cross_attention_ops(module, inputs, output):
    """Approximate MACs for channel cross-attention matmuls only."""
    x = inputs[0]
    if x.dim() != 4:
        return

    b, c, h, w = x.shape
    num_heads = module.num_heads
    c_per_head = c // num_heads
    hw = h * w

    qk_macs = b * num_heads * c_per_head * c_per_head * hw
    av_macs = b * num_heads * c_per_head * hw * c_per_head
    module.total_ops += torch.DoubleTensor([int(qk_macs + av_macs)])

# dummy 输入
net_for_profile = copy.deepcopy(net).to(device)
net_for_profile.eval()
total_params = count_parameters(net_for_profile, trainable_only=False)
trainable_params = count_parameters(net_for_profile, trainable_only=True)

# dummy 输入（和你之前一致）
dummy_LRs_RAW = torch.randn(1, 3, 4, 512, 512).to(device)
dummy_wb = torch.ones(1, 4).to(device)
dummy_cam2rgb = torch.ones(1, 3, 3).to(device)

# ===== Approx MACs / FLOPs / Params =====
macs, _ = profile(
    net_for_profile,
    inputs=(dummy_LRs_RAW, dummy_wb, dummy_cam2rgb),
    custom_ops={
        Attention: count_attention_ops,
        CrossAttention: count_cross_attention_ops,
    },
    verbose=False,
)
approx_flops = 2 * macs
total_params_fmt, trainable_params_fmt, macs_fmt, flops_fmt = clever_format(
    [total_params, trainable_params, macs, approx_flops], "%.3f"
)
print(
    f"[Stats] Total Params: {total_params_fmt}, "
    f"Trainable Params: {trainable_params_fmt}, "
    f"MACs: {macs_fmt}, Approx FLOPs: {flops_fmt}"
)
print("[Stats] Note: parameter counts are exact from model.parameters(); attention matmuls are approximated via custom_ops.")
print("[Stats] Note: flow_warp/grid_sample, softmax, normalize, and some functional ops may still be omitted by THOP.")

# ===== Inference Time（用副本或原 net 都可以）=====
with torch.no_grad():
    for _ in range(5):
        _ = net_for_profile(dummy_LRs_RAW, dummy_wb, dummy_cam2rgb)
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(30):
        _ = net_for_profile(dummy_LRs_RAW, dummy_wb, dummy_cam2rgb)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[Timing] Average inference time: {(time.time() - start)/30*1000:.3f} ms per batch")

# 加载测试数据集
test_dataloader = DataLoader(
    HDRVideoDataset(opt, 'test'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_workers
)
print(f"测试集样本总数：{len(test_dataloader)}")

# 加载训练好的权重
weight_path = os.path.join(opt.weight_savepath, f'{opt.model}/best.pth')
# weight_path = os.path.join(opt.weight_savepath, f'{opt.model}/checkpoints/epoch_62.pth')
weights = torch.load(weight_path)
net.load_state_dict(weights['model_state'])
print(f'成功加载HDR模型权重: {weight_path}')

net.eval()
hdr_psnr_list = []
hdr_ssim_list = []
hdr_psnr_mu_list = []  
ms_ssim_list = []

ldr_psnr_list = []
ldr_ssim_list = []
ldr_psnr_mu_list = []
ldr_ms_ssim_list = []


def mu_tonemap(hdr_image, mu=5000):
    # hdr_image 已经做了归一化（0~1 之间），这里直接算 μ 映射
    return torch.log(1 + mu * hdr_image) / torch.log(torch.tensor(1.0 + mu, device=hdr_image.device))


def norm_mu_tonemap(hdr_image, mu=5000, min_val=0.0):
    """
    对单帧 HDR 图 [B, 3, H, W] 做动态归一化 + μ 映射，同时对归一化后张量 clamp 到 non-negative。
    如果 max_val 不为 None，可以同时做上限限制（避免 mu*normalized 过大）。
    """
    B, C, H, W = hdr_image.shape
    pixel_max = hdr_image.max(dim=1)[0].view(B, -1)
    p_values  = torch.quantile(pixel_max, 0.997, dim=1).clamp(min=1e-10).view(B,1,1,1)
    normalized = hdr_image / p_values
    # 把所有负值都 clamp 回 0，保证做 log 时 (1 + mu*normalized) >= 1
    normalized = torch.clamp(normalized, min=min_val)

    return mu_tonemap(normalized, mu)

def preprocess_for_log_metrics(gt, pred, eps=1e-6, normalize=True):
    """
    针对 torch.Tensor 的 per-sample 预处理（用于 PSNR-L / SSIM-L）。
    gt, pred: torch.Tensor, shape [B,C,H,W] or [C,H,W]
    返回与输入同 device/dtype 的张量。
    normalize=True -> 每个样本分别除以该样本的 joint_max（gt_log 和 pred_log 的最大值）
    """
    # unify shape to [B, C, H, W]
    single = False
    if gt.dim() == 3:
        single = True
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)

    B = gt.shape[0]
    device = gt.device

    gt_log = torch.log1p(torch.clamp(gt, min=0.0) + eps)
    pred_log = torch.log1p(torch.clamp(pred, min=0.0) + eps)

    if normalize:
        # per-sample max: (B,)
        # flatten per-sample then max
        gt_flat = gt_log.view(B, -1)
        pred_flat = pred_log.view(B, -1)
        # joint_max per sample
        joint_max = torch.max(gt_flat.max(dim=1).values, pred_flat.max(dim=1).values)
        joint_max = joint_max.clamp(min=eps)  # avoid 0
        # shape broadcast to [B,1,1,1]
        joint_max = joint_max.view(B, 1, 1, 1).to(device)
        gt_log = gt_log / joint_max
        pred_log = pred_log / joint_max

    if single:
        return gt_log.squeeze(0), pred_log.squeeze(0)
    return gt_log, pred_log


def save_visualization(pred, gt, ldr_stack, mask_u, mask_o, img_name, save_dir, wb, cam2rgb):
    """
    保存可视化对比图像和EXR文件
    - pred, gt: [C, H, W] (单帧)
    - ldr_stack: [3, C, H, W] (3帧堆叠)
    """
    # 增加 batch 维度
    pred = pred.unsqueeze(0)   # [1, C, H, W]
    gt = gt.unsqueeze(0)     # [1, C, H, W]
    # ldr_stack 已经是 [3, C, H, W]，可以当作一个 batch=3 的张量处理
    
    # 创建场景子目录
    scene_name = os.path.basename(os.path.dirname(os.path.dirname(img_name)))
    frame_name = os.path.basename(os.path.dirname(img_name))
    save_path = os.path.join(save_dir, scene_name, frame_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 创建EXR保存目录
    exr_save_path = os.path.join('hdr_results', 'exr_outputs', scene_name, frame_name)
    os.makedirs(exr_save_path, exist_ok=True)

    # # 保存mask
    # mask_u_vis = (mask_u[0].cpu().numpy() * 255).astype(np.uint8)
    # mask_o_vis = (mask_o[0].cpu().numpy() * 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(save_path, 'mask_u_pred.png'), mask_u_vis)
    # cv2.imwrite(os.path.join(save_path, 'mask_o_pred.png'), mask_o_vis)

    # 第一步：生成线性HDR数据并保存为EXR（不使用tonemapping）
    out_hdr_linear = process_exr(pred.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), 
                                 gamma=1.0, use_demosaic=False, use_tonemapping=False, data_range=8.0)[0].numpy().transpose((1,2,0))
    gt_hdr_linear = process_exr(gt.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), 
                                 gamma=1.0, use_demosaic=False, use_tonemapping=False, data_range=8.0)[0].numpy().transpose((1,2,0))

    
    # 保存EXR文件（线性HDR数据）
    cv2.imwrite(os.path.join(exr_save_path, 'hdr_pred.exr'), 
                np.clip(out_hdr_linear, 0, None).astype(np.float32),
                [int(cv2.IMWRITE_EXR_TYPE), cv2.IMWRITE_EXR_TYPE_HALF])
    cv2.imwrite(os.path.join(exr_save_path, 'hdr_gt.exr'), 
                np.clip(gt_hdr_linear, 0, None).astype(np.float32),
                [int(cv2.IMWRITE_EXR_TYPE), cv2.IMWRITE_EXR_TYPE_HALF])


    # 第二步：生成tonemapped数据用于PNG可视化
    out_hdr = process(pred.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), 
                      gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
    gt_hdr = process(gt.cpu().detach(), wbs=wb[None].cpu(), cam2rgbs=cam2rgb[None].cpu(), 
                      gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
    
    # --- LDR 处理修改 (放弃广播，逐帧处理) ---
    # ldr_stack 是 [3, C, H, W]
    # wb 和 cam2rgb 是 [C_wb] 和 [3,3]
    # 我们将逐帧调用 process，每次都传入 [1, C, H, W] 的图像和 [1, ...] 的元数据
    
    # 1. 处理 LDR Prev (EV-2)
    in_ldr_prev_processed = process(
        ldr_stack[0].unsqueeze(0).cpu().detach(), # [1, C, H, W]
        wbs=wb[None].cpu(), 
        cam2rgbs=cam2rgb[None].cpu(), 
        gamma=2.2, 
        use_demosaic=False, 
        use_tonemapping=False, 
        data_range=1.0
    ) # 输出 [1, 3, H, W]

    # 2. 处理 LDR Center (EV0)
    in_ldr_center_processed = process(
        ldr_stack[1].unsqueeze(0).cpu().detach(), # [1, C, H, W]
        wbs=wb[None].cpu(), 
        cam2rgbs=cam2rgb[None].cpu(), 
        gamma=2.2, 
        use_demosaic=False, 
        use_tonemapping=False, 
        data_range=1.0
    ) # 输出 [1, 3, H, W]

    # 3. 处理 LDR Next (EV+2)
    in_ldr_next_processed = process(
        ldr_stack[2].unsqueeze(0).cpu().detach(), # [1, C, H, W]
        wbs=wb[None].cpu(), 
        cam2rgbs=cam2rgb[None].cpu(), 
        gamma=2.2, 
        use_demosaic=False, 
        use_tonemapping=False, 
        data_range=1.0
    ) # 输出 [1, 3, H, W]

    # 转换为 numpy 格式
    # (squeeze(0) 移除 batch 维度, transpose 转为 H,W,C)
    in_ldr_prev_np = (in_ldr_prev_processed.squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    in_ldr_center_np = (in_ldr_center_processed.squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    in_ldr_next_np = (in_ldr_next_processed.squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

    out_hdr = (out_hdr*255).astype(np.uint8)
    gt_hdr = (gt_hdr*255).astype(np.uint8)

    # 保存PNG可视化图像
    # 保存全部三帧 LDR
    cv2.imwrite(os.path.join(save_path, 'input_ldr_prev(EV0).png'), in_ldr_prev_np)
    cv2.imwrite(os.path.join(save_path, 'input_ldr_center(EV0).png'), in_ldr_center_np)
    cv2.imwrite(os.path.join(save_path, 'input_ldr_next(EV0).png'), in_ldr_next_np)
    
    cv2.imwrite(os.path.join(save_path, 'pred_hdr.png'), out_hdr)
    cv2.imwrite(os.path.join(save_path, 'gt_hdr.png'), gt_hdr)



for test_data in tqdm(test_dataloader):
    with torch.no_grad():
        # 加载HDR测试数据
        LRs_RAW = test_data['LDRs_RAW'].cuda()
        LRs_RAW_stack = LRs_RAW.clone() 
        # LRs_RAW_nopack = test_data['LDRs_unpacked'].cuda()
        HR_HDR_gt = test_data['HDR_DATA'].cuda()
        wb = test_data['wb'].to(device, non_blocking=True)
        cam2rgb = test_data['cam2rgb'].to(device, non_blocking=True)
        
        # 模型推理 + 颜色校正
        pred_HDR, mask_u, mask_o = net(LRs_RAW, wb, cam2rgb)

        LRs_RAW_center = LRs_RAW[:, LRs_RAW.shape[1] // 2]
        
        # 计算HDR质量指标
        # 对GT和Pred先预处理
        gt_log, pred_log = preprocess_for_log_metrics(HR_HDR_gt, pred_HDR)
        hdr_psnr_list.append(utils.get_psnr(gt_log, pred_log))
        hdr_ssim_list.append(utils.get_ssim(gt_log, pred_log))
        mu_img1 = norm_mu_tonemap(HR_HDR_gt, 5000)
        mu_img2 = norm_mu_tonemap(pred_HDR, 5000)
        hdr_psnr_mu_list.append(utils.get_psnr(mu_img1, mu_img2))
        # hdr_psnr_list.append(utils.get_psnr(HR_HDR_gt, pred_HDR))
        # hdr_ssim_list.append(utils.get_ssim(HR_HDR_gt, pred_HDR))
        
        msssim_per_sample = ms_ssim(mu_img1, mu_img2, data_range=1.0, size_average=False)
        msssim_val = msssim_per_sample.item()  # 转成 Python float
        ms_ssim_list.append(msssim_val)

        mu_input = norm_mu_tonemap(LRs_RAW_center, 5000)
        ldr_psnr_mu_list.append(utils.get_psnr(mu_img1, mu_input))
        ldr_psnr_list.append(utils.get_psnr(HR_HDR_gt, LRs_RAW_center))
        ldr_ssim_list.append(utils.get_ssim(HR_HDR_gt, LRs_RAW_center))
        ldr_messsim_per_sample = ms_ssim(mu_img1, mu_input, data_range=1.0, size_average=False)
        ldr_messsim_val = ldr_messsim_per_sample.item()
        ldr_ms_ssim_list.append(ldr_messsim_val)

        # 保存HDR结果（EXR格式）及可视化对比
        if opt.save_image:
            # 创建可视化目录
            vis_dir = os.path.join('hdr_results', 'visualization')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 获取批次数据
            batch_size = pred_HDR.shape[0]
            for idx in range(batch_size):
                # 一次性保存所有可视化PNG和EXR文件
                save_visualization(
                    pred_HDR[idx], 
                    HR_HDR_gt[idx], 
                    LRs_RAW_stack[idx],
                    mask_u[idx],
                    mask_o[idx],
                    img_name=test_data['HDR_gt_name'][idx],
                    save_dir=vis_dir,
                    wb=wb[idx],
                    cam2rgb=cam2rgb[idx]
                )
                


# 输出最终指标
avg_psnr = sum(hdr_psnr_list) / len(hdr_psnr_list)
avg_ssim = sum(hdr_ssim_list) / len(hdr_ssim_list)
avg_psnr_mu = sum(hdr_psnr_mu_list) / len(hdr_psnr_mu_list)
avg_ms_ssim = sum(ms_ssim_list) / len(ms_ssim_list)

ldr_avg_psnr = sum(ldr_psnr_list) / len(ldr_psnr_list)
ldr_avg_ssim = sum(ldr_ssim_list) / len(ldr_ssim_list)
ldr_avg_psnr_mu = sum(ldr_psnr_mu_list) / len(ldr_psnr_mu_list)
ldr_avg_ms_ssim = sum(ldr_ms_ssim_list) / len(ldr_ms_ssim_list)

print(f'HDR测试结果 PSNR-L: {avg_psnr:.2f} dB | PSNR-mu: {avg_psnr_mu:.2f} | SSIM-L: {avg_ssim:.4f} | MS-SSIM: {avg_ms_ssim:.4f}')
print(f'LDR测试结果 PSNR: {ldr_avg_psnr:.2f} dB | PSNR-mu: {ldr_avg_psnr_mu:.2f} | SSIM: {ldr_avg_ssim:.4f} | MS-SSIM: {ldr_avg_ms_ssim:.4f}')
