import os
import torch
import torchvision.utils as torchutils
import torch.nn.functional as F
import math
import torch.nn as nn
from pytorch_msssim import ssim


class l1_loss():
    def __init__(self, weight=1.0):
        self.weight = weight
        self.per_channel = None

    def __call__(self, pred, target):
        diff = F.l1_loss(pred, target, reduction='none')
        self.per_channel = diff.mean(dim=(0, 2, 3))
        total_loss = self.per_channel.mean() * self.weight
        return total_loss


class log_l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(torch.log(torch.clamp(pred, min=0) + 1 / 5000), torch.log(target + 1 / 5000))


class SimpleOverexposureColorLoss(nn.Module):
    def __init__(self, weight=0.3, smooth_weight=0.1, overexposure_weight=2.0):
        super().__init__()
        self.weight = weight
        self.smooth_weight = smooth_weight
        self.overexposure_weight = overexposure_weight

    def forward(self, pred_hdr, target_hdr, mask_over=None):
        overexposure_loss = 0
        if mask_over is not None:
            if mask_over.shape[1] == 1:
                mask_over = mask_over.repeat(1, pred_hdr.shape[1], 1, 1)

            masked_diff = torch.abs(pred_hdr - target_hdr) * mask_over
            mask_sum = torch.sum(mask_over, dim=[2, 3], keepdim=True) + 1e-8
            overexposure_loss = torch.mean(torch.sum(masked_diff, dim=[2, 3], keepdim=True) / mask_sum)

        grad_x = torch.abs(pred_hdr[:, :, :, 1:] - pred_hdr[:, :, :, :-1])
        grad_y = torch.abs(pred_hdr[:, :, 1:, :] - pred_hdr[:, :, :-1, :])
        smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)

        total_loss = self.overexposure_weight * overexposure_loss + self.smooth_weight * smooth_loss
        return self.weight * total_loss


class L1_ulaw(nn.Module):
    def __init__(self, mu: float = 5000.0):
        super().__init__()
        self.mu = float(mu)
        self._den = float(math.log1p(self.mu))

    def _tonemap(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=0.0)
        return torch.log1p(self.mu * x) / self._den

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        t_pred = self._tonemap(pred)
        t_gt = self._tonemap(gt)
        return torch.mean(torch.abs(t_pred - t_gt))

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
