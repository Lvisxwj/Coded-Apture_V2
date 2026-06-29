"""
train.py — Unified training script for vegetation index models.

Usage:
  python train.py                          # 读 cfg.py + config.yaml
  python train.py --config other.yaml      # 用别的配置文件
"""

import argparse
import datetime
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cfg import MODEL_NAME, GPU_ID
from dataset_npy import VegIdxDataset
from loss import CharbonnierLoss, SSIM
from Norm import max_min_norm


# ----------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------

def build_model(name, cfg):
    """Instantiate model from config."""
    model_cfg = cfg.get(name, {})

    if name == "mst_mamba":
        from models.mst_mamba import MST_Mamba_VegIdx
        return MST_Mamba_VegIdx(
            dim=model_cfg.get("dim", 64),
            stage=model_cfg.get("stage", 3),
            num_blocks=model_cfg.get("num_blocks", [2, 2, 2]),
            d_state=model_cfg.get("d_state", 16),
            expand_factor=model_cfg.get("expand_factor", 2),
            d_conv=model_cfg.get("d_conv", 4),
            dt_rank=model_cfg.get("dt_rank", "auto"),
            pscan_parallel=model_cfg.get("pscan_parallel", True),
            mask_fusion=model_cfg.get("mask_fusion", "input_mul"),
            stage_reductions=model_cfg.get("stage_reductions", None),
        )
    elif name == "wpo3d":
        from models.wpo3d import WPO3D_VegIdx
        return WPO3D_VegIdx(
            dim=model_cfg.get("dim", 64),
            stage=model_cfg.get("stage", 3),
            num_blocks=model_cfg.get("num_blocks", [2, 2, 2]),
        )
    elif name == "restormer":
        from models.restormer import Restormer_VegIdx
        return Restormer_VegIdx(
            dim=model_cfg.get("dim", 48),
            num_heads=model_cfg.get("num_heads", [1, 2, 4, 8]),
            num_blocks=model_cfg.get("num_blocks", [4, 6, 6, 8]),
            ffn_expansion_factor=model_cfg.get("ffn_expansion_factor", 2.66),
        )
    elif name == "dhm":
        from models.dhm import DHM_VegIdx
        return DHM_VegIdx(
            dim=model_cfg.get("dim", 64),
            stage=model_cfg.get("stage", 3),
            num_blocks=model_cfg.get("num_blocks", [2, 2, 2]),
            window_size=model_cfg.get("window_size", 8),
            d_state=model_cfg.get("d_state", 16),
            expand_factor=model_cfg.get("expand_factor", 2),
            d_conv=model_cfg.get("d_conv", 4),
        )
    elif name == "ifgnet":
        from models.ifgnet import IFGNet
        return IFGNet(
            dim_spec=model_cfg.get("dim_spec", 32),
            dim_idx=model_cfg.get("dim_idx", 48),
            stage=model_cfg.get("stage", 3),
            num_blocks_spec=model_cfg.get("num_blocks_spec", [1, 1, 1]),
            num_blocks_idx=model_cfg.get("num_blocks_idx", [2, 2, 2]),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: mst_mamba, wpo3d, restormer, dhm, ifgnet")


# ----------------------------------------------------------------
# Input normalization
# ----------------------------------------------------------------

def normalize_input(inputs):
    """Normalize each sample independently to avoid batch-dependent scaling."""
    if inputs.ndim == 3:  # [B, H, W]
        dims = (1, 2)
    elif inputs.ndim == 4:  # [B, C, H, W], independently per channel
        dims = (2, 3)
    else:
        raise ValueError(f"Expected a 3D or 4D tensor, got {tuple(inputs.shape)}")
    mn = inputs.amin(dim=dims, keepdim=True)
    mx = inputs.amax(dim=dims, keepdim=True)
    return (inputs - mn) / (mx - mn).clamp_min(1e-8)


def max_min_norm_diff(x):
    """Compatibility alias for the centralized differentiable normalizer."""
    return max_min_norm(x)


# ----------------------------------------------------------------
# PSNR
# ----------------------------------------------------------------

def compute_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10 * torch.log10(data_range ** 2 / mse)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VegIdx Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = MODEL_NAME
    device = torch.device(GPU_ID)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    batch_size = train_cfg.get("batch_size", 1)
    max_epoch = train_cfg.get("max_epoch", 300)
    lr = train_cfg.get("lr", 1e-4)
    train_set_size = train_cfg.get("train_set_size", 2560)
    grad_clip = train_cfg.get("gradient_clip", 0.2)
    save_interval = train_cfg.get("save_interval", 50)
    log_interval = train_cfg.get("log_interval", 50)
    output_dir = paths_cfg.get("output_dir", "./output/")
    use_amp = train_cfg.get("amp", False)
    use_compile = train_cfg.get("compile", False)

    # IFGNet config
    is_ifgnet = model_name == "ifgnet"
    if is_ifgnet:
        ifg_cfg = cfg.get("ifgnet", {})
        formula_warmup = ifg_cfg.get("formula_loss_warmup", 100)
        formula_weight = ifg_cfg.get("formula_loss_weight", 0.1)

    # Seed
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Dataset — build a mini-cfg dict compatible with VegIdxDataset
    ds_cfg = {
        "data": {
            "root": paths_cfg["data_root"],
            "hsi_dir": paths_cfg.get("hsi_dir", "HSI"),
            "label_dir": paths_cfg.get("label_dir", "label"),
            "mask_dir": paths_cfg.get("mask_dir", "mask"),
            **{k: v for k, v in data_cfg.items() if k != "augmentation"},
            "augmentation": data_cfg.get("augmentation", {}),
        },
        "train": train_cfg,
    }
    dataset = VegIdxDataset(ds_cfg, is_train=True)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )

    # Model
    model = build_model(model_name, cfg)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if use_compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, falling back: {e}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg.get("milestones", [200, 400]),
        gamma=train_cfg.get("gamma", 0.5),
    )

    # Loss functions
    loss_Charbonnier = CharbonnierLoss()
    loss_L1 = torch.nn.L1Loss()
    loss_SSIM = SSIM(val_range=1.0)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Output dirs
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, f"train_log_{model_name}.txt")

    print(f"Model: {model_name} | Params: {param_count:,} | Device: {device}")
    print(f"Batch: {batch_size} | LR: {lr} | Epochs: {max_epoch} | AMP: {use_amp}")
    print(f"Output: {model_dir}")

    # Training loop
    all_losses = []
    all_psnrs = []
    best_loss = float("inf")
    train_start_time = time.time()

    with open(log_file, "w") as f:
        f.write(f"Model: {model_name}, Params: {param_count:,}\n")
        f.write(f"LR: {lr}, Batch: {batch_size}, Epochs: {max_epoch}, AMP: {use_amp}\n")
        f.write(f"Started: {datetime.datetime.now()}\n\n")

        for epoch in range(1, max_epoch + 1):
            model.train()
            epoch_loss = 0.0
            epoch_charb = 0.0
            epoch_l1 = 0.0
            epoch_ssim = 0.0
            epoch_psnr = 0.0
            n_batches = 0
            t0 = time.time()

            for i, (hsi, inputs, labels, mask) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    inputs_norm = normalize_input(inputs)

                    if is_ifgnet:
                        I_final, I_formula, R_hat = model(inputs_norm, mask)
                        outputs = I_final
                    else:
                        outputs = model(inputs_norm, mask)

                    labels_norm = max_min_norm_diff(labels).detach()

                    charb = loss_Charbonnier(outputs, labels_norm)
                    l1 = loss_L1(outputs, labels_norm)
                    ssim_l = loss_SSIM(outputs, labels_norm)
                    loss = charb + l1 + ssim_l

                    if is_ifgnet and epoch > formula_warmup:
                        I_formula_norm = max_min_norm_diff(I_formula)
                        L_formula = F.l1_loss(outputs, I_formula_norm.detach())
                        loss = loss + formula_weight * L_formula

                # Guard: skip batch if loss is non-finite
                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss={loss.item():.4f} at E{epoch} B{i}, skipping")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Guard: check grads before clip (inf * 0 = NaN in clip_grad_norm_)
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_bad_grad:
                    print(f"WARNING: non-finite grad at E{epoch} B{i}, skipping step")
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    psnr_val = compute_psnr(outputs, labels_norm).item()

                epoch_loss += loss.item()
                epoch_charb += charb.item()
                epoch_l1 += l1.item()
                epoch_ssim += ssim_l.item()
                epoch_psnr += psnr_val
                n_batches += 1

                if i % log_interval == 0:
                    log = (
                        f"E{epoch:4d} B{i:4d}/{train_set_size // batch_size:4d} "
                        f"loss={epoch_loss / n_batches:.8f} "
                        f"charb={epoch_charb / n_batches:.8f} "
                        f"l1={epoch_l1 / n_batches:.8f} "
                        f"ssim={epoch_ssim / n_batches:.8f} "
                        f"psnr={epoch_psnr / n_batches:.2f}dB\n"
                    )
                    f.write(log)
                    f.flush()
                    print(log.strip())

            scheduler.step()

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_psnr = epoch_psnr / max(n_batches, 1)
            all_losses.append(avg_loss)
            all_psnrs.append(avg_psnr)

            epoch_log = (
                f"Epoch {epoch:4d} | loss={avg_loss:.8f} | psnr={avg_psnr:.2f}dB "
                f"| lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s\n"
            )
            f.write(epoch_log)
            f.flush()
            print(epoch_log.strip())

            # ETA every 50 epochs
            if epoch % 50 == 0:
                total_elapsed = time.time() - train_start_time
                epochs_done = epoch
                epochs_left = max_epoch - epoch
                avg_sec = total_elapsed / epochs_done
                eta_time = datetime.datetime.now() + datetime.timedelta(seconds=avg_sec * epochs_left)
                eta_log = (
                    f"  >> ETA: {epochs_left} epochs left, "
                    f"~{avg_sec:.1f}s/epoch, "
                    f"est. finish: {eta_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(eta_log)
                f.flush()
                print(eta_log.strip())

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(model_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "psnr": avg_psnr,
                }, best_path)
                best_log = f"  * New best: loss={avg_loss:.8f} psnr={avg_psnr:.2f}dB -> {best_path}\n"
                f.write(best_log)
                f.flush()
                print(best_log.strip())

            # Save periodic checkpoint
            if epoch % save_interval == 0 or epoch == max_epoch:
                ckpt_path = os.path.join(
                    model_dir,
                    f"{model_name}_ep{epoch}_lr{lr:.0e}_bs{batch_size}.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "psnr": avg_psnr,
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

        total_time = time.time() - train_start_time
        f.write(f"\nTotal training time: {total_time:.1f}s ({total_time/3600:.2f}h)\n")
        f.write(f"Best loss: {best_loss:.8f}\n")
        f.write(f"Completed: {datetime.datetime.now()}\n")

    # Loss & PSNR curves
    epochs_range = range(1, len(all_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(epochs_range, all_losses, "b-", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, all_psnrs, "r-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title(f"{model_name} Training PSNR")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{model_name}_training_curves.png"), dpi=200)
    plt.close()

    print(f"Training complete. Total time: {total_time/3600:.2f}h | Best loss: {best_loss:.8f}")


if __name__ == "__main__":
    main()
