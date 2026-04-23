import os
import torch
import torchvision.utils as torchutils
import torch.nn.functional as F
import math
from pytorch_msssim import ssim

def get_psnr(HR_gt, HR):
    HR_gt = HR_gt.detach().clone()
    HR = HR.detach().clone()
    diff = (HR - HR_gt).pow(2).mean() + 1e-8
    psnr = -10 * math.log10(diff)
    return psnr

def get_ssim(HR_gt, HR):
    HR_gt = HR_gt.detach().clone()
    HR = HR.detach().clone()
    ssim_all = ssim(HR_gt, HR, data_range=1, size_average=True)
    return ssim_all
