"""CPU-only smoke checks for the active vegindex pipeline.

This script never loads project data or checkpoints. It uses tiny synthetic
inputs to verify imports, tensor shapes, finite outputs and utility invariants.
"""

import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from Norm import max_min_norm, reverse_max_min_norm
from test import pad_to_patch_multiple, simulate_cassi
from train import build_model, normalize_input
from viz import stitch_patches


MODEL_CONFIG = {
    "mst_mamba": {
        "dim": 8, "stage": 1, "num_blocks": [1], "d_state": 4,
        "expand_factor": 1, "d_conv": 3, "pscan_parallel": False,
    },
    "wpo3d": {"dim": 8, "stage": 1, "num_blocks": [1]},
    "restormer": {
        "dim": 8, "num_heads": [1, 2], "num_blocks": [1, 1],
        "ffn_expansion_factor": 2.0,
    },
    "dhm": {
        "dim": 8, "stage": 1, "num_blocks": [1], "window_size": 4,
        "d_state": 4, "expand_factor": 1, "d_conv": 3,
    },
    "ifgnet": {
        "dim_spec": 8, "dim_idx": 8, "stage": 1,
        "num_blocks_spec": [1], "num_blocks_idx": [1],
    },
}


def check_utilities():
    # Per-sample normalization must not share extrema across the batch.
    x = torch.tensor([[[0.0, 1.0]], [[10.0, 20.0]]])
    normalized = normalize_input(x)
    assert torch.allclose(normalized.amin((1, 2)), torch.zeros(2))
    assert torch.allclose(normalized.amax((1, 2)), torch.ones(2))

    normalized_seed = torch.rand(2, 32, 3, 5)
    original = reverse_max_min_norm(normalized_seed)
    renormalized = max_min_norm(original)
    assert torch.allclose(normalized_seed, renormalized, rtol=1e-5, atol=1e-5)

    hsi = np.ones((17, 19, 84), dtype=np.float32)
    padded, pad_h, pad_w = pad_to_patch_multiple(hsi, 16)
    assert padded.shape == (32, 32, 84)
    assert (pad_h, pad_w) == (15, 13)

    measurement, mask = simulate_cassi(
        np.ones((16, 16, 84), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    )
    assert tuple(measurement.shape) == (16, 182)
    assert tuple(mask.shape) == (84, 16, 16)
    assert torch.isfinite(measurement).all() and torch.isfinite(mask).all()

    patch_a = np.ones((32, 16, 16), dtype=np.float32)
    patch_b = np.full((32, 16, 16), 2.0, dtype=np.float32)
    stitched = stitch_patches([(0, 0, patch_a), (0, 16, patch_b)], 16)
    assert stitched.shape == (32, 16, 32)
    assert np.all(stitched[:, :, :16] == 1.0)
    assert np.all(stitched[:, :, 16:] == 2.0)
    return {"utilities": "passed"}


def check_model(model_name):
    cfg = {model_name: MODEL_CONFIG[model_name]}
    torch.manual_seed(0)
    model = build_model(model_name, cfg).cpu().eval()
    measurement = torch.rand(1, 16, 182)
    mask = torch.rand(1, 84, 16, 16)
    with torch.inference_mode():
        output = model(measurement, mask)
        if model_name == "ifgnet":
            output = output[0]
    assert tuple(output.shape) == (1, 32, 16, 16), tuple(output.shape)
    assert torch.isfinite(output).all()
    return {
        "model": model_name,
        "shape": list(output.shape),
        "finite": True,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIG) + ["all"],
        default="all",
    )
    parser.add_argument("--utilities-only", action="store_true")
    args = parser.parse_args()

    results = [check_utilities()]
    if not args.utilities_only:
        names = list(MODEL_CONFIG) if args.model == "all" else [args.model]
        for name in names:
            results.append(check_model(name))
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
