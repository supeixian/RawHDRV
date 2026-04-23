from distutils.errors import DistutilsByteCompileError
from math import gamma
from this import d
from matplotlib.pyplot import axis
import rawpy
# import imageio
import cv2
import numpy as np
import torch
import exifread
import os

def center_crop(img, x, y):
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _ = img.shape
    xx = (h - x) // 2
    yy = (w - y) // 2
    return img[xx:xx+x, yy:yy+y]
 
def pack_raw_bayer(bayer_raw, raw_pattern, bl, wl):
    #pack Bayer image to 4 channels
    im = bayer_raw
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    white_point = wl
    black_level = bl
    
    # out = out - black_level
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def apply_gains_rgb(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    H, W, C = bayer_images.shape
    if C == 3:
        wbs = wbs[:3]
    outs = bayer_images * wbs[::-1].reshape(1, 1, C)
    return outs

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    # print(bayer_images.shape, wbs.shape)
    # print(wbs)
    if C == 3:
        wbs = wbs[:, :, :3]
    wbs = wbs.squeeze(1)
    # print(wbs.shape)
    # wbs = wbs[:, [2,1,0,3]]  # R和B交换，G1和G2不动
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    assert images.dim() == 4, "输入图像必须是 BxCxHxW 格式"
    B, C_in, H, W = images.shape
    assert ccms.shape[-2:] == (C_in, C_in), "CCM最后两维需为 (C_in, C_in)"
    
    # 调整维度顺序：BxCxHxW -> BxHxWxC
    images_permuted = images.permute(0, 2, 3, 1)  # [B, H, W, C]
    # 添加广播维度：BxHxWx1xC
    images_expanded = images_permuted.unsqueeze(-2)  # [B, H, W, 1, C]
    # 调整CCM维度：Bx...xCxC -> Bx1x1xCxC
    ccms_expanded = ccms.view(B, 1, 1, C_in, C_in)  # [B, 1, 1, C, C]
    # 矩阵乘法等效实现：逐元素乘后求和
    outputs = torch.sum(images_expanded * ccms_expanded, dim=-1)  # [B, H, W, C]
    # 恢复原始维度顺序：BxHxWxC -> BxCxHxW
    outputs = outputs.permute(0, 3, 1, 2)  # [B, C, H, W]

    return outputs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, 1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    # outs = np.clip((outs*255).int(), 0, 255).float() / 255
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,2,...], 
        torch.mean(bayer_images[:, [1,3], ...], dim=1), 
        bayer_images[:,0,...]], dim=1)

    return lin_rgb

def demosaic(in_vid, converter=cv2.COLOR_BayerBG2BGR):
    bayer_input = np.zeros([in_vid.shape[0], in_vid.shape[2] * 2, in_vid.shape[3] * 2], dtype=np.float32)
    bayer_input[:, ::2, ::2] = in_vid[:, 2, :, :]
    bayer_input[:, ::2, 1::2] = in_vid[:, 1, :, :]
    bayer_input[:, 1::2, ::2] = in_vid[:, 3, :, :]
    bayer_input[:, 1::2, 1::2] = in_vid[:, 0, :, :]
    bayer_input = (bayer_input * 65535).astype('uint16')
    rgb_input = np.zeros([bayer_input.shape[0], bayer_input.shape[1], bayer_input.shape[2], 3], dtype=np.float32)
    for j in range(bayer_input.shape[0]):
        rgb_input[j] = cv2.cvtColor(bayer_input[j], converter)
        # rgb_input[j] = demosaicing_CFA_Bayer_DDFAPD(bayer_input[j], 'BGGR')
    rgb_input = rgb_input.transpose((0, 3, 1, 2))
    rgb_input = torch.from_numpy(rgb_input/65535.)
    return rgb_input

def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.
    Args:
        hdr_image (np.ndarray or torch.Tensor): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        torch.Tensor: Mu-law tonemapped image.
    """
    # 确保 hdr_image 是浮点类型
    if isinstance(hdr_image, torch.Tensor):
        hdr_image = hdr_image.float()  # 转换为 Float
    else:
        hdr_image = torch.tensor(hdr_image, dtype=torch.float32)
    
    # 确保 mu 与 hdr_image 设备一致
    device = hdr_image.device
    mu_tensor = torch.tensor([mu], device=device, dtype=torch.float32)
    
    # 执行计算
    return torch.log(1 + mu_tensor * hdr_image) / torch.log(1 + mu_tensor)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.
    """
    return mu_tonemap(hdr_image/norm_value, mu)

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.
        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.
        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.
        """
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def process(bayer_images, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    """numpy.ndarray"""
    # device = torch.device('cpu')
    # bayer_images = bayer_images.to(device)
    # wbs = wbs.to(device)
    # cam2rgbs = cam2rgbs.to(device)
    # White balance.
    # print(torch.min(bayer_images), torch.max(bayer_images))
    bayer_images = apply_gains(bayer_images, wbs)
    # print(torch.min(bayer_images), torch.max(bayer_images))
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=data_range)
    if use_demosaic:
        images = demosaic(bayer_images)
    elif bayer_images.shape[1] == 4:
        images = binning(bayer_images)
    else:
        images = bayer_images
    # Color correction.
    # print(torch.min(images), torch.max(images))
    images = apply_ccms(images, cam2rgbs)
    # print(torch.min(images), torch.max(images))
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=data_range)
    if use_tonemapping:
        images = tanh_norm_mu_tonemap(images, data_range, 200)
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    # print(torch.min(images), torch.max(images))
    images = torch.clamp(images, min=0.0, max=1.0)
    return images

def process_exr(bayer_images, wbs, cam2rgbs, gamma=1.0, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """
    Processes a batch of Bayer RGBG images.
    (已修正)
    """
    # --- 1. 初始处理 (White Balance, Clamping) ---
    bayer_images = apply_gains(bayer_images, wbs)
    bayer_images = torch.clamp(bayer_images, min=0.0)
    
    # --- 2. Demosaic/Binning ---
    if use_demosaic:
        images = demosaic(bayer_images)
    elif bayer_images.shape[1] == 4:
        images = binning(bayer_images)
    else:
        images = bayer_images
        
    # --- 3. Color Correction ---
    images = apply_ccms(images, cam2rgbs)
    
    # --- 4. 最终输出处理 (关键修正) ---
    
    # 检查是否用于线性EXR输出
    # (根据 save_visualization 的调用: gamma=1.0 且 use_tonemapping=False)
    is_linear_exr_output = (gamma == 1.0) and (use_tonemapping == False)

    if is_linear_exr_output:
        # === A. EXR 输出分支 ===
        # 裁剪到 [0, data_range] (允许高光，去除负值)
        # 注意：您在 apply_gains 和 apply_ccms 之后可能也需要clamp
        images = torch.clamp(images, min=0.0) 
        # 直接返回线性数据，不进行gamma或[0,1]裁剪
        return images
    
    else:
        # === B. PNG (可视化) 输出分支 ===
        
        # 裁剪到 [0, data_range]
        images = torch.clamp(images, min=0.0, max=data_range) 

        if use_tonemapping:
            # 应用色调映射 (输出范围 [0, 1])
            images = tanh_norm_mu_tonemap(images, data_range, 200)
        else:
            # 如果不用色调映射，也必须压缩到 [0, 1] 以便后续gamma
            images = torch.clamp(images, min=0.0, max=1.0)

        # 应用 Gamma
        images = gamma_compression(images, gamma)
        
        # 最终裁剪到 [0, 1]
        images = torch.clamp(images, min=0.0, max=1.0)
        
        return images

def process_sequence(bayer_seq, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=1.0):
    """
    bayer_seq: [B, T, 4, H, W]
    wbs: [B, T, 3] 或可广播
    cam2rgbs: [B, T, 3, 3] 或可广播
    返回: Tensor [B, T, 3, H, W]
    """
    B, T, C4, H, W = bayer_seq.shape
    # 将 [B,T,4,H,W] -> [B*T,4,H,W]
    flat = bayer_seq.view(B*T, C4, H, W)
    # 同理 flatten wb 和 cam2rgbs到 [B*T, ...]
    # 假设 wbs shape [B,T,3]:
    wbs_flat = wbs.view(B*T, 3)
    # cam2rgbs: [B,T,3,3] -> [B*T,3,3]
    cam_flat = cam2rgbs.view(B*T, 3, 3)
    # 调用 process
    proc_flat = process(flat, wbs_flat, cam_flat, gamma, use_demosaic, use_tonemapping, data_range)  # [B*T,3,H,W]
    # reshape回 [B,T,3,H,W]
    proc_seq = proc_flat.view(B, T, 3, H, W)
    return proc_seq

def process_tiff(bayer_images, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    """numpy.ndarray"""
    # White balance.
    # print(torch.min(bayer_images), torch.max(bayer_images))
    bayer_images = apply_gains(bayer_images, wbs)
    # print(torch.min(bayer_images), torch.max(bayer_images))
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=data_range)
    if use_demosaic:
        images = demosaic(bayer_images)
    elif bayer_images.shape[1] == 4:
        images = binning(bayer_images)
    else:
        images = bayer_images
    # Color correction.
    # print(torch.min(images), torch.max(images))
    # print(cam2rgbs.shape)
    images = apply_ccms(images, cam2rgbs)
    # print(torch.min(images), torch.max(images))
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=data_range)
    if use_tonemapping:
        images = tanh_norm_mu_tonemap(images, data_range, 200)
    images = torch.clamp(images, min=0.0, max=data_range)
    # images = gamma_compression(images, gamma)
    # print(torch.min(images), torch.max(images))
    # images = torch.clamp(images, min=0.0, max=1.0)
    return images

def get_cam2rgb(xyz2cam):
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    # rgb2xyz = np.eye(3)
    rgb2cam = np.dot(xyz2cam, rgb2xyz)
    norm = np.tile(np.sum(rgb2cam, 1), (3, 1)).transpose()
    rgb2cam = rgb2cam / norm
    cam2rgb = np.linalg.inv(rgb2cam)
    return cam2rgb

def get_isp_params(raw):
    bayer_2by2 = raw.raw_pattern
    wb = np.array(raw.camera_whitebalance) 
    wb = np.array([float(wb[0]), float(wb[1]), float(wb[2]), float(wb[1])], dtype=np.float32)
    wb /= wb[1]
    xyz2cam = raw.color_matrix[:3, :3].astype(np.float32)
    row_norms = np.linalg.norm(xyz2cam, axis=1, keepdims=True)
    cam2rgb = xyz2cam / np.where(row_norms > 0, row_norms, 1.0)
    # cam2rgb = get_cam2rgb(xyz2cam)
    pattern = np.array(bayer_2by2)
    black_level = raw.black_level_per_channel[0] 
    white_level = raw.white_level
    return pattern, wb, black_level, white_level, cam2rgb

def read_raw(path):
    with rawpy.imread(path) as raw:
        bayer_raw = raw.raw_image_visible.astype(np.float32)
        pattern, wb, black_level, white_level, cam2rgb = get_isp_params(raw)
        packed_raw = pack_raw_bayer(bayer_raw, pattern, raw.black_level_per_channel[0], raw.white_level)
    return packed_raw, pattern, wb, black_level, white_level, cam2rgb

def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo
# def color_correction(bayer_images, wb, cam2rgb):
#     image = bayer_images * wb
#     """RGBG -> RGB"""
#     rgb = np.stack([bayer_images[0, :, :], np.mean(bayer_images[[1,3], ...], axis=0), bayer_images[2, ...]], axis=0)

#     return rgb
def pack_raw_bayer_v2(bayer_raw, raw_pattern, bl, wl):
    #pack Bayer image to 4 channels
    im = bayer_raw
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    white_point = wl
    black_level = bl
    
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out