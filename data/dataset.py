import os
import re
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from rawhdr.process import unpack_raw_bayer, read_raw
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class HDRVideoDataset(Dataset):
    """
    优化策略：
    1. 默认禁用Dataset内的线程池，依赖DataLoader的多进程并行
    2. 可选启用小规模线程池用于加速单样本内的多帧读取
    3. 添加数据缓存减少重复I/O
    """
    def __init__(self, opt, mode):
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.N_frames = opt.N_frames
        self.half_N = self.N_frames // 2
        self.raw_exposure = f"EV_{opt.target_exposure}.dng"
        self.scene_frames = self._build_scene_index()
        
        # 线程池配置：默认禁用（use_threading=False）
        self.use_threading = getattr(opt, 'use_threading', False)
        if self.use_threading:
            # 如果启用，使用小规模线程池（2-4个worker）
            max_workers = getattr(opt, 'thread_workers', 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            # print(f"[Dataset] 线程池已启用，workers={max_workers}")
        else:
            self.executor = None
            # print("[Dataset] 线程池已禁用，使用顺序读取")
        
        # 元数据缓存（轻量级，可以保留）
        self._metadata_cache = {}

    def _build_scene_index(self):
        if self.mode == 'test':
            data_root = self.opt.test_root
            scenes = sorted(glob.glob(os.path.join(data_root, 'scene_*')))
        else:
            data_root = self.opt.train_root
            all_scenes = sorted(glob.glob(os.path.join(data_root, 'scene_*')))
            
            if self.mode == 'train_val':
                np.random.seed(42)
                np.random.shuffle(all_scenes)
                split_idx = int(len(all_scenes) * 0.95)
                scenes = all_scenes[split_idx:]
            else:
                scenes = all_scenes
                
        scene_data = []
        
        for scene_path in scenes:
            frames = sorted([
                f for f in glob.glob(os.path.join(scene_path, 'frame_*'))
                if os.path.exists(os.path.join(f, self.raw_exposure)) and 
                os.path.exists(os.path.join(f, 'hdr.mat'))
            ], key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
            
            valid_sequences = []
            for frame_idx, frame_dir in enumerate(frames):
                start = max(0, frame_idx - self.half_N)
                end = min(len(frames), frame_idx + self.half_N + 1)
                seq_frames = frames[start:end]
                
                while len(seq_frames) < self.N_frames:
                    if start == 0:
                        seq_frames = [frames[0]] * (self.N_frames - len(seq_frames)) + seq_frames
                    else:
                        seq_frames = seq_frames + [frames[-1]] * (self.N_frames - len(seq_frames))
                
                valid_sequences.append((
                    [os.path.join(f, self.raw_exposure) for f in seq_frames],
                    [os.path.join(f, 'hdr.mat') for f in seq_frames],
                    frame_idx - start
                ))
            scene_data.extend(valid_sequences)
        return scene_data

    def _load_metadata(self, mat_path):
        """缓存元数据"""
        if mat_path in self._metadata_cache:
            return self._metadata_cache[mat_path]
        
        mat_data = sio.loadmat(mat_path, variable_names=['wb', 'pattern', 'cam2rgb'])
        metadata = (mat_data['wb'][0], mat_data['pattern'], mat_data['cam2rgb'])
        self._metadata_cache[mat_path] = metadata
        return metadata

    def _load_data_sequential(self, raw_paths, mat_paths, center_idx):
        """顺序读取数据（推荐用于HDD和训练）"""
        # 顺序读取RAW文件
        ldr_sequence = []
        for path in raw_paths:
            ldr_sequence.append(read_raw(path)[0])
        ldr_sequence = np.stack(ldr_sequence, axis=0)
        
        # 读取HDR
        mat_data = sio.loadmat(mat_paths[center_idx], variable_names=['hdr'])
        hdr = mat_data['hdr']
        
        # 读取元数据
        metadata_list = [self._load_metadata(mp) for mp in mat_paths]
        
        return ldr_sequence, hdr, metadata_list

    def _load_data_parallel(self, raw_paths, mat_paths, center_idx):
        """并行读取数据（可选用于SSD和测试）"""
        # 并行读取RAW文件
        raw_futures = [self.executor.submit(read_raw, path) for path in raw_paths]
        
        # 并行读取HDR
        hdr_future = self.executor.submit(
            sio.loadmat, 
            mat_paths[center_idx], 
            variable_names=['hdr']
        )
        
        # 并行读取元数据
        meta_futures = [self.executor.submit(self._load_metadata, mp) for mp in mat_paths]
        
        # 收集结果
        ldr_sequence = [f.result()[0] for f in raw_futures]
        ldr_sequence = np.stack(ldr_sequence, axis=0)
        hdr = hdr_future.result()['hdr']
        metadata_list = [f.result() for f in meta_futures]
        
        return ldr_sequence, hdr, metadata_list

    def crop_data(self, data, coords=None, size=None):
        """裁剪数据"""
        if size is None:
            return data
        
        if len(data.shape) == 4:
            T, C, H, W = data.shape
            is_sequence = True
        else:
            C, H, W = data.shape
            is_sequence = False
           
        crop_w, crop_h = size
        crop_w = min(crop_w, W)
        crop_h = min(crop_h, H)
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)
       
        if coords is None:
            x = (W - crop_w) // 2
            y = (H - crop_h) // 2
        else:
            x, y = coords
            x = min(max(0, x), W - crop_w)
            y = min(max(0, y), H - crop_h)
       
        x = x - (x % 2)
        y = y - (y % 2)
       
        if is_sequence:
            return data[:, :, y:y+crop_h, x:x+crop_w]
        else:
            return data[:, y:y+crop_h, x:x+crop_w]

    def __getitem__(self, index):
        raw_paths, mat_paths, center_idx = self.scene_frames[index]
        
        # 根据配置选择读取方式
        if self.use_threading and self.executor is not None:
            ldr_sequence, hdr, metadata_list = self._load_data_parallel(
                raw_paths, mat_paths, center_idx
            )
        else:
            ldr_sequence, hdr, metadata_list = self._load_data_sequential(
                raw_paths, mat_paths, center_idx
            )
        
        # 裁剪逻辑
        H, W = ldr_sequence.shape[2], ldr_sequence.shape[3]
        
        if self.mode == 'train':
            crop_size = self.opt.crop_size - (self.opt.crop_size % 2)
            x = np.random.randint(0, (W - crop_size)//2) * 2
            y = np.random.randint(0, (H - crop_size)//2) * 2
            
            ldr_sequence = ldr_sequence[:, :, y:y+crop_size, x:x+crop_size]
            hdr = hdr[:, y:y+crop_size, x:x+crop_size] if hdr is not None else None
            
        elif self.mode == 'test':
            crop_coords = getattr(self.opt, 'crop_coords', None)
            crop_size_custom = getattr(self.opt, 'crop_size_custom', None)
            
            if crop_size_custom is not None:
                ldr_sequence = self.crop_data(ldr_sequence, crop_coords, crop_size_custom)
                if hdr is not None:
                    hdr = self.crop_data(hdr, crop_coords, crop_size_custom)
            else:
                crop_size = min(self.opt.crop_size, H, W)
                crop_size = crop_size - (crop_size % 2)
                x = (W - crop_size) // 2
                y = (H - crop_size) // 2
                x = x - (x % 2)
                y = y - (y % 2)
                
                ldr_sequence = ldr_sequence[:, :, y:y+crop_size, x:x+crop_size]
                hdr = hdr[:, y:y+crop_size, x:x+crop_size] if hdr is not None else None
        else:
            crop_size = min(self.opt.crop_size, H, W)
            crop_size = crop_size - (crop_size % 2)
            x = (W - crop_size) // 2
            y = (H - crop_size) // 2
            x = x - (x % 2)
            y = y - (y % 2)
            
            ldr_sequence = ldr_sequence[:, :, y:y+crop_size, x:x+crop_size]
            hdr = hdr[:, y:y+crop_size, x:x+crop_size] if hdr is not None else None
        
        # 解包LDR数据
        ldr_unpacked = unpack_raw_bayer(ldr_sequence, metadata_list[center_idx][1])
        
        # 获取中心帧元数据
        wb, pattern, cam2rgb = metadata_list[center_idx]
        
        return {
            'LDRs_RAW': torch.FloatTensor(ldr_sequence),
            'LDRs_unpacked': torch.FloatTensor(ldr_unpacked),
            'HDR_DATA': torch.FloatTensor(hdr) if hdr is not None else None,
            'wb': torch.FloatTensor(wb),
            'cam2rgb': torch.FloatTensor(cam2rgb),
            'bayer_pattern': torch.LongTensor(pattern),
            'HDR_gt_name': mat_paths[center_idx]
        }

    def __len__(self):
        return len(self.scene_frames)
    
    def __del__(self):
        """清理线程池"""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=False)