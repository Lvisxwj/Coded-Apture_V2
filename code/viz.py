"""
viz.py — Visualize test results: stitch patches into full leaves, draw per-index images.

Usage:
  1. Run test.py to get preds/*.npy and labels/*.npy
  2. Copy the test result directory path below
  3. python viz.py

Output:
  {TEST_RESULT_DIR}/viz/
  ├── file_0201/
  │   ├── 00_DD.png
  │   ├── 01_TVI.png
  │   └── ...
  ├── file_0202/
  │   └── ...
  └── ...
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# ── 修改这里来控制可视化 ──
# ================================================================

# test.py 输出的结果目录（粘贴路径到这里）
TEST_RESULT_DIR = ""        # ← 例: "/data5/.../result/test/restormer_ep300"

# 可视化参数
NUM_SAMPLES = -1            # -1 = 全部 test files，或指定数量（如 5）
INDICES = "all"             # "all" = 全部 32 个指数，或列表如 [0, 2, 12]
SHOW_ERROR_MAP = True       # True = 3列（label | pred | error），False = 2列（label | pred）
COLORMAP = "viridis"        # matplotlib colormap
DPI = 150

# ================================================================

# 32 个指数名称
INDEX_NAMES = [
    "DD", "TVI", "LCI", "mND680", "mND705", "PSSR", "CRI550", "CRI700",
    "MCARI", "SAVI", "CI_green", "CI_red_edge", "NDVI", "DVI", "ATSAVI",
    "EVI", "GI", "MSAVI", "MSR", "MVTI1", "MVTI2", "OSAVI", "PSND",
    "RDVI", "SPVI", "TCARI", "SR", "VARI_green", "WDRVI", "ARI", "BGI", "BRI",
]


def parse_patch_name(filename):
    """
    Parse 'file_0201_r000_c000.npy' -> (file_id=201, row=0, col=0).
    """
    m = re.match(r"file_(\d+)_r(\d+)_c(\d+)\.npy", filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def load_and_group_patches(npy_dir):
    """
    Load all .npy files from a directory, group by file_id.

    Returns:
        dict: {file_id: [(row, col, array), ...]}
    """
    groups = defaultdict(list)
    for fname in sorted(os.listdir(npy_dir)):
        if not fname.endswith(".npy"):
            continue
        parsed = parse_patch_name(fname)
        if parsed is None:
            continue
        fid, row, col = parsed
        arr = np.load(os.path.join(npy_dir, fname))  # [32, 256, 256]
        if arr.ndim != 3 or arr.shape[0] != 32:
            raise ValueError(f"Unexpected patch shape in {fname}: {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values found in {fname}")
        groups[fid].append((row, col, arr))
    return groups


def stitch_patches(patches, patch_size=256):
    """
    Stitch patches into a full image.

    Args:
        patches: [(row, col, array), ...] where array is [C, H, W]
        patch_size: size of each patch

    Returns:
        full_image: [C, full_H, full_W] numpy array
    """
    if not patches:
        return None

    C = patches[0][2].shape[0]
    max_r = max(r + arr.shape[1] for r, _, arr in patches)
    max_c = max(c + arr.shape[2] for _, c, arr in patches)

    full = np.zeros((C, max_r, max_c), dtype=np.float32)
    for r, c, arr in patches:
        full[:, r:r+patch_size, c:c+patch_size] = arr

    return full


def load_original_sizes(result_dir):
    """Load JSON metadata and return {file_id: (height, width)}."""
    meta_path = os.path.join(result_dir, "test_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"test_meta.json not found in {result_dir}; rerun the corrected test.py"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        int(item["file_id"]): (
            int(item["original_height"]), int(item["original_width"])
        )
        for item in meta.get("files", [])
    }


def draw_index_image(label_ch, pred_ch, index_name, save_path,
                     show_error=True, cmap="viridis", dpi=150):
    """
    Draw a single index comparison image.

    Args:
        label_ch: [H, W] numpy — ground truth for one index
        pred_ch: [H, W] numpy — prediction for one index
        index_name: str
        save_path: output path
        show_error: if True, add error map as 3rd column
        cmap: colormap name
        dpi: output dpi
    """
    vmin = min(label_ch.min(), pred_ch.min())
    vmax = max(label_ch.max(), pred_ch.max())
    if vmax <= vmin:
        vmax = vmin + 1e-8

    n_cols = 3 if show_error else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    im0 = axes[0].imshow(label_ch, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Label")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_ch, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if show_error:
        err = np.abs(label_ch - pred_ch)
        l1_val = np.mean(err)
        im2 = axes[2].imshow(err, cmap="hot", vmin=0, vmax=max(err.max(), 1e-8))
        axes[2].set_title(f"Error (L1={l1_val:.4f})")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(index_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    assert TEST_RESULT_DIR and os.path.isdir(TEST_RESULT_DIR), \
        f"Set TEST_RESULT_DIR at the top of viz.py. Current: '{TEST_RESULT_DIR}'"

    pred_dir = os.path.join(TEST_RESULT_DIR, "preds")
    label_dir = os.path.join(TEST_RESULT_DIR, "labels")
    viz_dir = os.path.join(TEST_RESULT_DIR, "viz")

    assert os.path.isdir(pred_dir), f"preds/ not found in {TEST_RESULT_DIR}"
    assert os.path.isdir(label_dir), f"labels/ not found in {TEST_RESULT_DIR}"
    original_sizes = load_original_sizes(TEST_RESULT_DIR)

    os.makedirs(viz_dir, exist_ok=True)

    # Determine which indices to visualize
    if INDICES == "all":
        idx_list = list(range(32))
    else:
        idx_list = list(INDICES)

    # Load and group patches
    print("Loading predictions...")
    pred_groups = load_and_group_patches(pred_dir)
    print("Loading labels...")
    label_groups = load_and_group_patches(label_dir)

    # Sort file IDs
    file_ids = sorted(pred_groups.keys())
    if NUM_SAMPLES > 0:
        file_ids = file_ids[:NUM_SAMPLES]

    print(f"Files: {len(file_ids)} | Indices: {len(idx_list)} | "
          f"Error map: {SHOW_ERROR_MAP} | Colormap: {COLORMAP}")

    for fi, fid in enumerate(file_ids):
        if fid not in label_groups:
            print(f"Warning: file_{fid:04d} has preds but no labels, skipping")
            continue

        # Stitch patches into full images
        pred_full = stitch_patches(pred_groups[fid])   # [32, H, W]
        label_full = stitch_patches(label_groups[fid]) # [32, H, W]

        if pred_full is None or label_full is None:
            continue
        if fid not in original_sizes:
            raise KeyError(f"Original dimensions missing for file_{fid:04d}")
        original_h, original_w = original_sizes[fid]
        pred_full = pred_full[:, :original_h, :original_w]
        label_full = label_full[:, :original_h, :original_w]

        # Create per-file output directory
        file_viz_dir = os.path.join(viz_dir, f"file_{fid:04d}")
        os.makedirs(file_viz_dir, exist_ok=True)

        for idx in idx_list:
            if idx >= pred_full.shape[0]:
                continue
            name = INDEX_NAMES[idx] if idx < len(INDEX_NAMES) else f"ch{idx}"
            save_path = os.path.join(file_viz_dir, f"{idx:02d}_{name}.png")

            draw_index_image(
                label_full[idx], pred_full[idx],
                index_name=f"{name} (file {fid:04d})",
                save_path=save_path,
                show_error=SHOW_ERROR_MAP,
                cmap=COLORMAP,
                dpi=DPI,
            )

        print(f"[{fi+1}/{len(file_ids)}] file_{fid:04d}: "
              f"{pred_full.shape[1]}x{pred_full.shape[2]}, "
              f"{len(idx_list)} indices -> {file_viz_dir}")

    print(f"\nVisualization complete: {viz_dir}")


if __name__ == "__main__":
    main()
