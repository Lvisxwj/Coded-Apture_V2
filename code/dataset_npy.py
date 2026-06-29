"""
dataset_npy.py — Dataset loader for npy-format CASSI data.

Data layout:
  {root}/HSI/    — hsi_0001.npy ~ hsi_0252.npy, each [H, W, 84] float32 HSI cubes
  {root}/label/  — label_0001.npy ~ label_0252.npy, each [H, W, 32] float32 vegetation indices
  {root}/mask/   — mask.npy, single [256, 256] float32 CASSI mask

Returns per sample:
  hsi:           [84, 256, 256]   — cropped HSI patch
  measurement:   [256, 422]       — simulated CASSI 2D measurement
  target:        [32, 256, 256]   — vegetation index ground truth
  mask_3d_shift: [84, 256, 256]   — shifted 3D mask (cropped, no extra 422 dim on GPU)
"""

import os
import random

import numpy as np
import torch
import torch.utils.data as tud


class VegIdxDataset(tud.Dataset):
    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg: dict loaded from config.yaml
            is_train: True for training set, False for test set
        """
        super().__init__()
        data_cfg = cfg["data"]
        self.size = data_cfg.get("spatial_size", 256)
        self.num_bands = data_cfg.get("num_bands", 84)
        self.step = data_cfg.get("step", 2)
        self.meas_scale = data_cfg.get("measurement_scale", 0.9)
        self.is_train = is_train
        self.lazy_load = cfg.get("train", {}).get("lazy_load", False) if is_train \
            else cfg.get("test", {}).get("lazy_load", False)

        root = data_cfg["root"]
        hsi_dir = os.path.join(root, data_cfg.get("hsi_dir", "HSI"))
        label_dir = os.path.join(root, data_cfg.get("label_dir", "label"))
        mask_dir = os.path.join(root, data_cfg.get("mask_dir", "mask"))

        # Determine file ID range
        if is_train:
            id_range = data_cfg.get("train_range", [1, 200])
        else:
            id_range = data_cfg.get("test_range", [201, 252])

        # Converted NPY samples are continuously renumbered. Raw-MAT corrupt
        # IDs belong only in the legacy checkpoint configuration snapshot.
        skip_ids = set(data_cfg.get("skip_ids", []))

        # Virtual epoch size
        if is_train:
            self.epoch_size = cfg.get("train", {}).get("train_set_size", 2560)
        else:
            self.epoch_size = cfg.get("test", {}).get("test_set_size", 1200)

        # Augmentation config
        aug_cfg = data_cfg.get("augmentation", {})
        self.aug_rotation = aug_cfg.get("random_rotation", True) and is_train
        self.aug_hflip = aug_cfg.get("horizontal_flip", True) and is_train
        self.aug_vflip = aug_cfg.get("vertical_flip", True) and is_train

        # Collect file paths
        self.hsi_paths = []
        self.label_paths = []
        for fid in range(id_range[0], id_range[1] + 1):
            if fid in skip_ids:
                continue
            hsi_path = os.path.join(hsi_dir, f"hsi_{fid:04d}.npy")
            label_path = os.path.join(label_dir, f"label_{fid:04d}.npy")
            if not os.path.exists(hsi_path) or not os.path.exists(label_path):
                print(f"Warning: skipping file_id {fid} (file not found)")
                continue
            self.hsi_paths.append(hsi_path)
            self.label_paths.append(label_path)

        assert len(self.hsi_paths) > 0, f"No data found in {hsi_dir}"

        # Load data: eager (all in RAM) or lazy (memory-mapped)
        if self.lazy_load:
            self.data = None
            self.labels = None
            print(f"Lazy loading enabled: {len(self.hsi_paths)} samples ({'train' if is_train else 'test'})")
        else:
            self.data = []
            self.labels = []
            for i, (hp, lp) in enumerate(zip(self.hsi_paths, self.label_paths)):
                print(f"Loading {i+1}/{len(self.hsi_paths)}...")
                self.data.append(np.load(hp))
                self.labels.append(np.load(lp))
            print(f"Loaded {len(self.data)} samples ({'train' if is_train else 'test'})")

        # Load mask
        mask_path = os.path.join(mask_dir, "mask.npy")
        if os.path.exists(mask_path):
            self.mask = np.load(mask_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        self.mask_h, self.mask_w = self.mask.shape
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, self.num_bands))

    def _load_sample(self, index):
        """Load a single sample (lazy or eager)."""
        if self.lazy_load:
            hsi = np.load(self.hsi_paths[index], mmap_mode='r')
            label = np.load(self.label_paths[index], mmap_mode='r')
        else:
            hsi = self.data[index]
            label = self.labels[index]
        return hsi, label

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        nC = self.num_bands
        step = self.step

        # Random sample selection
        n_files = len(self.hsi_paths)
        index = random.randint(0, n_files - 1)
        hsi, target = self._load_sample(index)

        H, W, _ = hsi.shape

        # Random spatial crop
        px = random.randint(0, H - self.size)
        py = random.randint(0, W - self.size)
        hsi = np.array(hsi[px:px + self.size, py:py + self.size, :])
        target = np.array(target[px:px + self.size, py:py + self.size, :])

        # Random mask crop
        pxm = random.randint(0, self.mask_h - self.size)
        pym = random.randint(0, self.mask_w - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size, pym:pym + self.size, :]

        # Data augmentation
        if self.is_train:
            if self.aug_rotation:
                rot_times = random.randint(0, 3)
                for _ in range(rot_times):
                    hsi = np.rot90(hsi)
                    target = np.rot90(target)

            if self.aug_vflip and random.random() > 0.5:
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()

            if self.aug_hflip and random.random() > 0.5:
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        # Simulate CASSI measurement (vectorized — no per-band np.roll)
        temp = mask_3d * hsi  # [size, size, nC]
        meas_width = self.size + (nC - 1) * step
        temp_shift = np.zeros((self.size, meas_width, nC), dtype=np.float32)
        mask_3d_shift = np.zeros((self.size, meas_width, nC), dtype=np.float32)
        for t in range(nC):
            offset = step * t
            temp_shift[:, offset:offset + self.size, t] = temp[:, :, t]
            mask_3d_shift[:, offset:offset + self.size, t] = mask_3d[:, :, t]

        measurement = np.sum(temp_shift, axis=2)  # [size, meas_width]
        measurement = measurement / nC * self.meas_scale

        # Only keep cropped mask [size, size, nC] — avoid sending full 422-wide mask to GPU
        mask_3d_cropped = mask_3d_shift[:, :self.size, :]

        # Convert to tensors — channels first
        measurement = torch.FloatTensor(measurement.copy())                                 # [256, 422]
        hsi_t = torch.FloatTensor(hsi.astype(np.float32).copy()).permute(2, 0, 1)          # [84, 256, 256]
        target_t = torch.FloatTensor(target.astype(np.float32).copy()).permute(2, 0, 1)     # [32, 256, 256]
        mask_t = torch.FloatTensor(mask_3d_cropped.copy()).permute(2, 0, 1)                 # [84, 256, 256]

        return hsi_t, measurement, target_t, mask_t
