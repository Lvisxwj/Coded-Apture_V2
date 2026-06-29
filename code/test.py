"""
test.py — Deterministic testing with sliding window inference.

Usage:
  python test.py                          # 读 cfg.py + config.yaml
  python test.py --config other.yaml      # 用别的配置文件

Output:
  test_dir/{model_name}/
  ├── test_log.txt                        # per-channel 指标表格
  ├── per_channel_metrics.png             # L1/PSNR bar chart
  ├── metrics_summary.csv                 # 结构化指标
  ├── preds/
  │   ├── file_0201_r000_c000.npy         # [32, 256, 256] 原始尺度预测
  │   └── ...
  └── labels/
      ├── file_0201_r000_c000.npy         # [32, 256, 256] 原始尺度标签
      └── ...
"""

import argparse
import datetime
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from cfg import MODEL_NAME, GPU_ID, BEST_CKPT, INDEX_NAMES
from train import build_model, normalize_input, max_min_norm_diff
from Norm import max_min_norm, reverse_max_min_norm


# ----------------------------------------------------------------
# Per-channel metrics
# ----------------------------------------------------------------

def compute_per_channel_l1(pred, target):
    """pred, target: [C, H, W] numpy, in [0,1] normalized space."""
    return [np.mean(np.abs(pred[c] - target[c])) for c in range(pred.shape[0])]


def compute_per_channel_psnr(pred, target, data_range=1.0):
    psnrs = []
    for c in range(pred.shape[0]):
        mse = np.mean((pred[c] - target[c]) ** 2)
        if mse < 1e-10:
            psnrs.append(100.0)
        else:
            psnrs.append(10 * np.log10(data_range ** 2 / mse))
    return psnrs


# ----------------------------------------------------------------
# CASSI measurement simulation (matches dataset_npy.py)
# ----------------------------------------------------------------

def simulate_cassi(hsi_patch, mask_2d, step=2, scale=0.9):
    """
    Simulate CASSI 2D measurement from an HSI patch.

    Args:
        hsi_patch: [H, W, nC] numpy float32
        mask_2d: [H, W] numpy float32 (will be tiled to 3D)
        step: dispersion step
        scale: measurement normalization scale

    Returns:
        measurement: [H, meas_width] tensor float32
        mask_3d_cropped: [nC, H, W] tensor float32 (cropped to spatial size)
    """
    if hsi_patch.ndim != 3:
        raise ValueError(f"hsi_patch must be [H, W, C], got {hsi_patch.shape}")
    if mask_2d.ndim != 2:
        raise ValueError(f"mask_2d must be [H, W], got {mask_2d.shape}")
    if not np.isfinite(hsi_patch).all() or not np.isfinite(mask_2d).all():
        raise ValueError("CASSI input contains NaN or Inf")
    H, W, nC = hsi_patch.shape
    if mask_2d.shape[0] < H or mask_2d.shape[1] < W:
        raise ValueError(
            f"Mask {mask_2d.shape} is smaller than patch {(H, W)}"
        )
    mask_3d = np.tile(mask_2d[:H, :W, np.newaxis], (1, 1, nC))

    temp = mask_3d * hsi_patch
    meas_width = W + (nC - 1) * step
    temp_shift = np.zeros((H, meas_width, nC), dtype=np.float32)
    mask_3d_shift = np.zeros((H, meas_width, nC), dtype=np.float32)
    for t in range(nC):
        offset = step * t
        temp_shift[:, offset:offset + W, t] = temp[:, :, t]
        mask_3d_shift[:, offset:offset + W, t] = mask_3d[:, :, t]

    measurement = np.sum(temp_shift, axis=2) / nC * scale
    mask_3d_cropped = mask_3d_shift[:, :W, :]

    measurement = torch.FloatTensor(measurement)               # [H, meas_width]
    mask_3d_cropped = torch.FloatTensor(mask_3d_cropped).permute(2, 0, 1)  # [nC, H, W]
    return measurement, mask_3d_cropped


def pad_to_patch_multiple(array, patch_size):
    """Reflect-pad an HWC array and return it with the applied padding."""
    if array.ndim != 3:
        raise ValueError(f"Expected HWC array, got {array.shape}")
    pad_h = (patch_size - array.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - array.shape[1] % patch_size) % patch_size
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return array, pad_h, pad_w


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VegIdx Testing")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = MODEL_NAME
    device = torch.device(GPU_ID)
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]

    ckpt_path = BEST_CKPT
    assert ckpt_path and os.path.exists(ckpt_path), \
        f"Set BEST_CKPT in cfg.py. Current: '{ckpt_path}'"

    nC = data_cfg.get("num_bands", 84)
    patch_size = data_cfg.get("spatial_size", 256)
    step = data_cfg.get("step", 2)
    meas_scale = data_cfg.get("measurement_scale", 0.9)
    test_range = data_cfg.get("test_range", [201, 252])
    skip_ids = set(data_cfg.get("skip_ids", []))

    # Paths
    data_root = paths_cfg["data_root"]
    hsi_dir = os.path.join(data_root, paths_cfg.get("hsi_dir", "HSI"))
    label_dir = os.path.join(data_root, paths_cfg.get("label_dir", "label"))
    mask_dir = os.path.join(data_root, paths_cfg.get("mask_dir", "mask"))
    test_output = paths_cfg.get("test_dir", "./test_results/")

    # Load mask
    mask_path = os.path.join(mask_dir, "mask.npy")
    mask_2d = np.load(mask_path).astype(np.float32)  # [256, 256]

    # Model
    model = build_model(model_name, cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_epoch = ckpt.get("epoch", 0)
    else:
        model = ckpt
        ckpt_epoch = 0
    model.to(device)
    model.eval()

    # Output directory
    run_name = f"{model_name}_ep{ckpt_epoch}"
    out_dir = os.path.join(test_output, run_name)
    pred_dir = os.path.join(out_dir, "preds")
    label_save_dir = os.path.join(out_dir, "labels")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    log_file = os.path.join(out_dir, f"test_log_{model_name}.txt")

    print(f"Testing {model_name} on {device}")
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt_epoch})")
    print(f"Output: {out_dir}")

    # Collect test files
    test_files = []
    for fid in range(test_range[0], test_range[1] + 1):
        if fid in skip_ids:
            continue
        hp = os.path.join(hsi_dir, f"hsi_{fid:04d}.npy")
        lp = os.path.join(label_dir, f"label_{fid:04d}.npy")
        if os.path.exists(hp) and os.path.exists(lp):
            test_files.append((fid, hp, lp))
        else:
            print(f"Warning: skipping file_id {fid} (not found)")

    print(f"Test files: {len(test_files)}")
    if not test_files:
        raise RuntimeError("No test files were found; check config paths and split")

    # Accumulators
    all_channel_l1 = []
    all_channel_psnr = []
    total_patches = 0
    file_metadata = []
    t0 = time.time()

    with open(log_file, "w") as f:
        f.write(f"Test: {model_name}\n")
        f.write(f"Checkpoint: {ckpt_path} (epoch {ckpt_epoch})\n")
        f.write(f"Started: {datetime.datetime.now()}\n\n")

        with torch.no_grad():
            for file_idx, (fid, hp, lp) in enumerate(test_files):
                hsi_full = np.load(hp).astype(np.float32)    # [H, W, 84]
                label_full = np.load(lp).astype(np.float32)  # [H, W, 32]
                H_orig, W_orig, _ = hsi_full.shape

                if not np.isfinite(hsi_full).all() or not np.isfinite(label_full).all():
                    raise ValueError(f"file_{fid:04d} contains NaN or Inf")

                # Pad to multiples of patch_size. Original dimensions are kept
                # in JSON so stitched visualizations can remove padded borders.
                hsi_full, pad_h, pad_w = pad_to_patch_multiple(hsi_full, patch_size)
                label_full, label_pad_h, label_pad_w = pad_to_patch_multiple(
                    label_full, patch_size
                )
                if (pad_h, pad_w) != (label_pad_h, label_pad_w):
                    raise RuntimeError(f"HSI/label padding mismatch for file_{fid:04d}")

                H_pad, W_pad = hsi_full.shape[0], hsi_full.shape[1]
                n_rows = H_pad // patch_size
                n_cols = W_pad // patch_size
                file_metadata.append({
                    "file_id": fid,
                    "original_height": H_orig,
                    "original_width": W_orig,
                    "padded_height": H_pad,
                    "padded_width": W_pad,
                    "pad_bottom": pad_h,
                    "pad_right": pad_w,
                    "patch_rows": n_rows,
                    "patch_cols": n_cols,
                })

                for ri in range(n_rows):
                    for ci in range(n_cols):
                        r0 = ri * patch_size
                        c0 = ci * patch_size
                        hsi_patch = hsi_full[r0:r0+patch_size, c0:c0+patch_size, :]
                        label_patch = label_full[r0:r0+patch_size, c0:c0+patch_size, :]

                        # Simulate CASSI measurement
                        measurement, mask_3d = simulate_cassi(
                            hsi_patch, mask_2d, step=step, scale=meas_scale
                        )

                        # To device, add batch dim
                        meas_b = measurement.unsqueeze(0).to(device)       # [1, H, meas_w]
                        mask_b = mask_3d.unsqueeze(0).to(device)           # [1, 84, H, W]

                        # Normalize input
                        meas_norm = normalize_input(meas_b)

                        # Forward
                        is_ifgnet = model_name == "ifgnet"
                        if is_ifgnet:
                            I_final, _, _ = model(meas_norm, mask_b)
                            output = I_final
                        else:
                            output = model(meas_norm, mask_b)

                        if output.shape != (1, 32, patch_size, patch_size):
                            raise RuntimeError(
                                f"Unexpected model output shape: {tuple(output.shape)}"
                            )
                        if not torch.isfinite(output).all():
                            raise FloatingPointError(
                                f"Non-finite output for file_{fid:04d} at ({r0}, {c0})"
                            )

                        # output is in [0,1] normalized space
                        out_norm_np = output[0].cpu().numpy()          # [32, H, W]

                        # Label normalized for metrics
                        label_t = torch.FloatTensor(label_patch.copy()).permute(2, 0, 1).unsqueeze(0)
                        label_norm = max_min_norm(label_t)
                        lab_norm_np = label_norm[0].numpy()            # [32, H, W]

                        # Per-channel metrics (normalized space)
                        all_channel_l1.append(compute_per_channel_l1(out_norm_np, lab_norm_np))
                        all_channel_psnr.append(compute_per_channel_psnr(out_norm_np, lab_norm_np))

                        # Reverse normalize for saving (original scale)
                        out_orig = reverse_max_min_norm(output.cpu()).numpy()[0]    # [32, H, W]
                        lab_orig = label_patch.transpose(2, 0, 1)                  # [32, H, W]

                        # Save npy
                        patch_name = f"file_{fid:04d}_r{r0:03d}_c{c0:03d}"
                        np.save(os.path.join(pred_dir, f"{patch_name}.npy"), out_orig)
                        np.save(os.path.join(label_save_dir, f"{patch_name}.npy"), lab_orig)

                        total_patches += 1

                f.write(f"File {fid:04d}: {H_orig}x{W_orig} -> {n_rows}x{n_cols} patches\n")
                print(f"[{file_idx+1}/{len(test_files)}] file_{fid:04d}: "
                      f"{H_orig}x{W_orig} -> {n_rows*n_cols} patches")

        elapsed = time.time() - t0

        # Average metrics
        mean_l1 = np.mean(all_channel_l1, axis=0)
        mean_psnr = np.mean(all_channel_psnr, axis=0)

        # Summary text
        summary = (
            f"\n{'='*60}\n"
            f"Test Results — {model_name} (epoch {ckpt_epoch})\n"
            f"{'='*60}\n"
            f"Files: {len(test_files)} | Patches: {total_patches} | Time: {elapsed:.1f}s\n"
            f"\n{'Index':<15} {'L1':>10} {'PSNR(dB)':>10}\n"
            f"{'-'*35}\n"
        )
        for c in range(32):
            name = INDEX_NAMES[c] if c < len(INDEX_NAMES) else f"ch{c}"
            summary += f"{name:<15} {mean_l1[c]:10.6f} {mean_psnr[c]:10.2f}\n"
        summary += f"{'-'*35}\n"
        summary += f"{'MEAN':<15} {np.mean(mean_l1):10.6f} {np.mean(mean_psnr):10.2f}\n"

        f.write(summary)
        print(summary)

        # Save CSV
        csv_path = os.path.join(out_dir, "metrics_summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as csvf:
            csvf.write("index,name,l1,psnr\n")
            for c in range(32):
                name = INDEX_NAMES[c] if c < len(INDEX_NAMES) else f"ch{c}"
                csvf.write(f"{c},{name},{mean_l1[c]:.8f},{mean_psnr[c]:.4f}\n")
            csvf.write(f"mean,MEAN,{np.mean(mean_l1):.8f},{np.mean(mean_psnr):.4f}\n")

        f.write(f"\nCompleted: {datetime.datetime.now()}\n")

    # Per-channel bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(32)

    ax1.bar(x, mean_l1, color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(INDEX_NAMES, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("L1 Error")
    ax1.set_title(f"{model_name} — Per-Channel L1")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, mean_psnr, color="coral", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(INDEX_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title(f"{model_name} — Per-Channel PSNR")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_per_channel_metrics.png"), dpi=200)
    plt.close()

    # Save transparent metadata for viz.py (no pickle required).
    meta = {
        "schema_version": 1,
        "model_name": model_name,
        "ckpt_epoch": ckpt_epoch,
        "checkpoint": ckpt_path,
        "patch_size": patch_size,
        "measurement_step": step,
        "measurement_scale": meas_scale,
        "files": file_metadata,
    }
    with open(os.path.join(out_dir, "test_meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"Patches: {total_patches} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
