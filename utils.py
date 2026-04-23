import os
import torch
import torchvision.utils as torchutils
import torch.nn.functional as F
import math
import cv2
import numpy as np
from pytorch_msssim import ssim

def save_checkpoints(state, is_best, save_dir):
    """Saves checkpoint to disk"""
    path = os.path.dirname(save_dir) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(state, save_dir)
    if is_best:
        torch.save(state, path + 'best.pth')


def save_RGB(image, scale, save_name, model_name):

    savepath = './results/RGB/' + scale + 'X/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_num = len(image)

    for idx in range(batch_num):
        torchutils.save_image(image[idx], savepath + os.path.basename(save_name[idx]).replace('HR', 'SR').replace('.png', '_{:s}.png'.format(model_name)))


def get_loss(out_im, gt_im, mask=None):
    if mask is None:
        return torch.abs(out_im - gt_im).mean()
    else:
        return torch.abs((out_im - gt_im) * mask).mean()


def get_CharbonnierLoss(out_im, gt_im, valid=None):
    if valid is None:
        diff = out_im - gt_im
        loss = torch.sqrt(diff * diff + 1e-6).mean()
        return loss
    else:
        diff = out_im - gt_im
        loss = torch.sqrt(diff * diff + 1e-6) * valid
        return loss.mean()


def get_mseloss(out_im, gt_im, valid=None):
    if valid is None:
        return F.mse_loss(out_im, gt_im)
    else:
        return F.mse_loss(out_im * valid, gt_im * valid)

def get_psnr(HR_gt, HR):
    HR_gt = HR_gt.detach().clone()
    HR = HR.detach().clone()
    diff = (HR - HR_gt).pow(2).mean() + 1e-8
    psnr = -10 * math.log10(diff)
    return psnr


def get_ssim(HR_gt, HR):
    # if HR_gt.dim() == 3:
    #     HR_gt = HR_gt.unsqueeze(0)
    # if HR.dim() == 3:
    #     HR = HR.unsqueeze(0)
    HR_gt = HR_gt.detach().clone()
    HR = HR.detach().clone()
    ssim_all = ssim(HR_gt, HR, data_range=1, size_average=True)
    return ssim_all

def pack_rggb_raw(raw):
    # pack RGGB Bayer raw to 4 channels
    _, _, H, W = raw.shape
    raw_pack = torch.cat((raw[:, :, 0:H:2, 0:W:2],
                          raw[:, :, 0:H:2, 1:W:2],
                          raw[:, :, 1:H:2, 0:W:2],
                          raw[:, :, 1:H:2, 1:W:2]), dim=1).cuda()
    return raw_pack

def depack_rggb_raw(raw):
    # depack 4 channels raw to RGGB Bayer
    _, H, W = raw.shape
    output = np.zeros((H * 2, W * 2))

    output[0:2 * H:2, 0:2 * W:2] = raw[0, :, :]
    output[0:2 * H:2, 1:2 * W:2] = raw[1, :, :]
    output[1:2 * H:2, 0:2 * W:2] = raw[2, :, :]
    output[1:2 * H:2, 1:2 * W:2] = raw[3, :, :]

    return output

def save_EXR(tensor,  img_names, save_dir='hdr_results'):
    """保存HDR结果为EXR格式（保留原始目录结构）"""
    for idx in range(tensor.shape[0]):
        # 获取原始文件路径信息
        original_path = os.path.dirname(img_names[idx])  # 原始目录结构（如：.../scene_01/frame_001）
        relative_path = os.path.relpath(original_path, start=os.path.commonpath([img_names[0], save_dir]))  # 提取相对路径
        
        # 构建保存路径
        save_path = os.path.join(save_dir, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        # 生成带序号的文件名（防止重名）
        frame_num = os.path.basename(original_path).split('_')[-1]
        save_name = f"hdr_{frame_num}.exr"
        
        # 转换并保存张量
        img = tensor[idx].permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, None)
        cv2.imwrite(os.path.join(save_path, save_name), 
                   img.astype(np.float32),
                   [int(cv2.IMWRITE_EXR_TYPE), cv2.IMWRITE_EXR_TYPE_HALF])