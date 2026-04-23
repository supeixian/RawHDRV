import os
import re
import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from data.process import read_raw


class HDRVideoDataset(Dataset):
    """HDR video dataset for RAW sequence reconstruction."""

    def __init__(self, opt, mode):
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.n_frames = int(opt.N_frames)
        self.half_n = self.n_frames // 2
        self.raw_exposure = f"EV_{opt.target_exposure}.dng"
        self.scene_frames = self._build_scene_index()

        self.use_threading = getattr(opt, 'use_threading', False)
        if self.use_threading:
            max_workers = getattr(opt, 'thread_workers', 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None

        self._metadata_cache = {}

    def _get_scenes(self):
        if self.mode == 'test':
            data_root = self.opt.test_root
            return sorted(glob.glob(os.path.join(data_root, 'scene_*')))

        data_root = self.opt.train_root
        all_scenes = sorted(glob.glob(os.path.join(data_root, 'scene_*')))

        if self.mode == 'train_val':
            rng = np.random.RandomState(42)
            rng.shuffle(all_scenes)
            split_idx = int(len(all_scenes) * 0.95)
            return all_scenes[split_idx:]

        return all_scenes

    @staticmethod
    def _frame_index(path):
        match = re.search(r'frame_(\d+)', path)
        return int(match.group(1)) if match else -1

    def _list_valid_frames(self, scene_path):
        frames = glob.glob(os.path.join(scene_path, 'frame_*'))
        frames = [
            f for f in frames
            if os.path.exists(os.path.join(f, self.raw_exposure))
            and os.path.exists(os.path.join(f, 'hdr.mat'))
        ]
        return sorted(frames, key=self._frame_index)

    def _build_sequence_window(self, frames, center_idx):
        start = max(0, center_idx - self.half_n)
        end = min(len(frames), center_idx + self.half_n + 1)
        seq_frames = frames[start:end]

        while len(seq_frames) < self.n_frames:
            if start == 0:
                seq_frames = [frames[0]] * (self.n_frames - len(seq_frames)) + seq_frames
            else:
                seq_frames = seq_frames + [frames[-1]] * (self.n_frames - len(seq_frames))

        center_in_seq = center_idx - start
        return seq_frames, center_in_seq

    def _build_scene_index(self):
        scenes = self._get_scenes()
        scene_data = []

        for scene_path in scenes:
            frames = self._list_valid_frames(scene_path)
            if not frames:
                continue

            for frame_idx, _ in enumerate(frames):
                seq_frames, center_in_seq = self._build_sequence_window(frames, frame_idx)
                scene_data.append((
                    [os.path.join(f, self.raw_exposure) for f in seq_frames],
                    [os.path.join(f, 'hdr.mat') for f in seq_frames],
                    center_in_seq,
                ))

        return scene_data

    def _load_metadata(self, mat_path):
        if mat_path in self._metadata_cache:
            return self._metadata_cache[mat_path]

        mat_data = sio.loadmat(mat_path, variable_names=['wb', 'pattern', 'cam2rgb'])
        metadata = (mat_data['wb'][0], mat_data['pattern'], mat_data['cam2rgb'])
        self._metadata_cache[mat_path] = metadata
        return metadata

    def _load_data_sequential(self, raw_paths, mat_paths, center_idx):
        ldr_sequence = [read_raw(path)[0] for path in raw_paths]
        ldr_sequence = np.stack(ldr_sequence, axis=0)

        mat_data = sio.loadmat(mat_paths[center_idx], variable_names=['hdr'])
        hdr = mat_data['hdr']
        metadata_list = [self._load_metadata(mp) for mp in mat_paths]

        return ldr_sequence, hdr, metadata_list

    def _load_data_parallel(self, raw_paths, mat_paths, center_idx):
        raw_futures = [self.executor.submit(read_raw, path) for path in raw_paths]
        hdr_future = self.executor.submit(sio.loadmat, mat_paths[center_idx], variable_names=['hdr'])
        meta_futures = [self.executor.submit(self._load_metadata, mp) for mp in mat_paths]

        ldr_sequence = [f.result()[0] for f in raw_futures]
        ldr_sequence = np.stack(ldr_sequence, axis=0)
        hdr = hdr_future.result()['hdr']
        metadata_list = [f.result() for f in meta_futures]

        return ldr_sequence, hdr, metadata_list

    @staticmethod
    def _even_crop_params(h, w, crop_h, crop_w, coords=None):
        crop_w = max(2, min(crop_w, w))
        crop_h = max(2, min(crop_h, h))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2

        if coords is None:
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
        else:
            x, y = coords
            x = min(max(0, x), w - crop_w)
            y = min(max(0, y), h - crop_h)

        x -= x % 2
        y -= y % 2
        return x, y, crop_w, crop_h

    def crop_data(self, data, coords=None, size=None):
        if size is None:
            return data

        if len(data.shape) == 4:
            _, _, h, w = data.shape
            is_sequence = True
        else:
            _, h, w = data.shape
            is_sequence = False

        crop_w, crop_h = size
        x, y, crop_w, crop_h = self._even_crop_params(h, w, crop_h, crop_w, coords=coords)

        if is_sequence:
            return data[:, :, y:y + crop_h, x:x + crop_w]
        return data[:, y:y + crop_h, x:x + crop_w]

    def _random_train_crop(self, ldr_sequence, hdr):
        h, w = ldr_sequence.shape[2], ldr_sequence.shape[3]
        crop = min(self.opt.crop_size, h, w)
        crop -= crop % 2

        max_x = max(0, w - crop)
        max_y = max(0, h - crop)
        x = (np.random.randint(0, max_x // 2 + 1) * 2) if max_x > 0 else 0
        y = (np.random.randint(0, max_y // 2 + 1) * 2) if max_y > 0 else 0

        ldr_sequence = ldr_sequence[:, :, y:y + crop, x:x + crop]
        if hdr is not None:
            hdr = hdr[:, y:y + crop, x:x + crop]
        return ldr_sequence, hdr

    def _default_center_crop(self, ldr_sequence, hdr):
        h, w = ldr_sequence.shape[2], ldr_sequence.shape[3]
        crop = min(self.opt.crop_size, h, w)
        crop -= crop % 2

        x, y, crop_w, crop_h = self._even_crop_params(h, w, crop, crop)
        ldr_sequence = ldr_sequence[:, :, y:y + crop_h, x:x + crop_w]
        if hdr is not None:
            hdr = hdr[:, y:y + crop_h, x:x + crop_w]
        return ldr_sequence, hdr

    def __getitem__(self, index):
        raw_paths, mat_paths, center_idx = self.scene_frames[index]

        if self.use_threading and self.executor is not None:
            ldr_sequence, hdr, metadata_list = self._load_data_parallel(raw_paths, mat_paths, center_idx)
        else:
            ldr_sequence, hdr, metadata_list = self._load_data_sequential(raw_paths, mat_paths, center_idx)

        if self.mode == 'train':
            ldr_sequence, hdr = self._random_train_crop(ldr_sequence, hdr)
        elif self.mode == 'test':
            crop_coords = getattr(self.opt, 'crop_coords', None)
            crop_size_custom = getattr(self.opt, 'crop_size_custom', None)

            if crop_size_custom is not None:
                ldr_sequence = self.crop_data(ldr_sequence, crop_coords, crop_size_custom)
                if hdr is not None:
                    hdr = self.crop_data(hdr, crop_coords, crop_size_custom)
            else:
                ldr_sequence, hdr = self._default_center_crop(ldr_sequence, hdr)
        else:
            ldr_sequence, hdr = self._default_center_crop(ldr_sequence, hdr)

        wb, pattern, cam2rgb = metadata_list[center_idx]

        return {
            'LDRs_RAW': torch.FloatTensor(ldr_sequence),
            'HDR_DATA': torch.FloatTensor(hdr) if hdr is not None else None,
            'wb': torch.FloatTensor(wb),
            'cam2rgb': torch.FloatTensor(cam2rgb),
            'bayer_pattern': torch.LongTensor(pattern),
            'HDR_gt_name': mat_paths[center_idx]
        }

    def __len__(self):
        return len(self.scene_frames)

    def __del__(self):
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=False)