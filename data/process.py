import rawpy
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
    """Apply white balance on BxCxHxW tensors."""
    N, C, _, _ = bayer_images.shape

    if wbs.dim() == 1:
        wbs = wbs.unsqueeze(0)
    if wbs.dim() == 3 and wbs.shape[1] == 1:
        wbs = wbs.squeeze(1)

    if C == 3:
        wbs = wbs[:, :3]

    if wbs.shape[0] != N and wbs.shape[0] == 1:
        wbs = wbs.expand(N, -1)

    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    """Apply color correction matrices on BxCxHxW tensors."""
    assert images.dim() == 4, 'Expected BxCxHxW images.'
    B, C_in, H, W = images.shape
    assert ccms.shape[-2:] == (C_in, C_in), 'CCM shape must end with (C, C).'

    if ccms.dim() == 2:
        ccms = ccms.unsqueeze(0)
    if ccms.shape[0] != B and ccms.shape[0] == 1:
        ccms = ccms.expand(B, -1, -1)

    images_hwc = images.permute(0, 2, 3, 1)
    outputs_hwc = torch.einsum('bij,bhwj->bhwi', ccms, images_hwc)
    return outputs_hwc.permute(0, 3, 1, 2)


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
    if isinstance(in_vid, torch.Tensor):
        in_vid = in_vid.detach().cpu().numpy()

    bayer_input = np.zeros([in_vid.shape[0], in_vid.shape[2] * 2, in_vid.shape[3] * 2], dtype=np.float32)
    bayer_input[:, ::2, ::2] = in_vid[:, 2, :, :]
    bayer_input[:, ::2, 1::2] = in_vid[:, 1, :, :]
    bayer_input[:, 1::2, ::2] = in_vid[:, 3, :, :]
    bayer_input[:, 1::2, 1::2] = in_vid[:, 0, :, :]
    bayer_input = (bayer_input * 65535).astype('uint16')
    rgb_input = np.zeros([bayer_input.shape[0], bayer_input.shape[1], bayer_input.shape[2], 3], dtype=np.float32)
    for j in range(bayer_input.shape[0]):
        rgb_input[j] = cv2.cvtColor(bayer_input[j], converter)
    rgb_input = rgb_input.transpose((0, 3, 1, 2))
    rgb_input = torch.from_numpy(rgb_input/65535.)
    return rgb_input

def mu_tonemap(hdr_image, mu=5000):
    """Compute mu-law tone mapping for linear HDR data."""
    if not isinstance(hdr_image, torch.Tensor):
        hdr_image = torch.tensor(hdr_image, dtype=torch.float32)
    else:
        hdr_image = hdr_image.float()

    mu_tensor = torch.as_tensor(mu, dtype=hdr_image.dtype, device=hdr_image.device)
    return torch.log1p(mu_tensor * hdr_image) / torch.log1p(mu_tensor)

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
    """Apply tanh normalization then mu-law tone mapping."""
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)


def _to_linear_rgb(bayer_images, wbs, cam2rgbs, use_demosaic=False, max_value=None):
    """Shared front-end: WB -> clamp -> demosaic/binning -> CCM."""
    bayer_images = apply_gains(bayer_images, wbs)
    if max_value is None:
        bayer_images = torch.clamp(bayer_images, min=0.0)
    else:
        bayer_images = torch.clamp(bayer_images, min=0.0, max=max_value)

    if use_demosaic:
        images = demosaic(bayer_images)
    elif bayer_images.shape[1] == 4:
        images = binning(bayer_images)
    else:
        images = bayer_images

    return apply_ccms(images, cam2rgbs)

def process(bayer_images, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """Convert RGBG tensors into display-ready sRGB in [0, 1]."""
    images = _to_linear_rgb(bayer_images, wbs, cam2rgbs, use_demosaic=use_demosaic, max_value=data_range)
    images = torch.clamp(images, min=0.0, max=data_range)
    if use_tonemapping:
        images = tanh_norm_mu_tonemap(images, data_range, 200)
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    return torch.clamp(images, min=0.0, max=1.0)

def process_exr(bayer_images, wbs, cam2rgbs, gamma=1.0, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """Process Bayer tensors for EXR (linear) or visualization outputs."""
    images = _to_linear_rgb(bayer_images, wbs, cam2rgbs, use_demosaic=use_demosaic, max_value=None)
    is_linear_exr_output = (gamma == 1.0) and (use_tonemapping == False)

    if is_linear_exr_output:
        return torch.clamp(images, min=0.0)

    images = torch.clamp(images, min=0.0, max=data_range)
    if use_tonemapping:
        images = tanh_norm_mu_tonemap(images, data_range, 200)
    else:
        images = torch.clamp(images, min=0.0, max=1.0)

    images = gamma_compression(images, gamma)
    return torch.clamp(images, min=0.0, max=1.0)

def process_sequence(bayer_seq, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=1.0):
    """Process Bayer sequence [B, T, 4, H, W] to [B, T, 3, H, W]."""
    B, T, C4, H, W = bayer_seq.shape
    flat = bayer_seq.view(B*T, C4, H, W)

    if wbs.dim() == 2:
        wbs = wbs.unsqueeze(1).expand(-1, T, -1)
    wbs_flat = wbs.reshape(B*T, -1)

    if cam2rgbs.dim() == 3:
        cam2rgbs = cam2rgbs.unsqueeze(1).expand(-1, T, -1, -1)
    cam_flat = cam2rgbs.reshape(B*T, 3, 3)

    proc_flat = process(flat, wbs_flat, cam_flat, gamma, use_demosaic, use_tonemapping, data_range)
    proc_seq = proc_flat.view(B, T, 3, H, W)
    return proc_seq

def process_tiff(bayer_images, wbs, cam2rgbs, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=8.0):
    """Process Bayer tensors for TIFF-like linear/ranged outputs."""
    images = _to_linear_rgb(bayer_images, wbs, cam2rgbs, use_demosaic=use_demosaic, max_value=data_range)
    images = torch.clamp(images, min=0.0, max=data_range)
    if use_tonemapping:
        images = tanh_norm_mu_tonemap(images, data_range, 200)
        return torch.clamp(images, min=0.0, max=1.0)
    return torch.clamp(images, min=0.0, max=data_range)

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

    return iso, expo

def pack_raw_bayer_v2(bayer_raw, raw_pattern, bl, wl):
    """Backward-compatible alias."""
    return pack_raw_bayer(bayer_raw, raw_pattern, bl, wl)