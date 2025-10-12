from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from MST_corrected import MST_Corrected
from mst_dataset_corrected import MST_dataset_corrected
import datetime
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import clip_grad_norm_

# Import loss functions and normalization (local copies)
from loss import CharbonnierLoss, SSIM
from Norm import max_min_norm

# Data paths (same as your original)
hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/cor_r_label/"

# Load dataset (EXACTLY like your original)
Dataset = MST_dataset_corrected(hsi_filename, label_filename)

# Model setup
model = MST_Corrected(dim=64, stage=3, num_blocks=[2,2,2])
Model_name = "MST_Corrected"
file_path = "/home/graduate/lizhuo/xwj/"
log_file = f"{file_path}parameter/ours/training_log_{Model_name}_lr4-4_bs8_Epoch500_2560_hybridLoss.txt"

# GPU setup (same as your original)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters (same as your original)
learning_rate = 0.0001
milestones = [200, 400]
gamma = 0.5
batch_size = 4
Max_epoch = 500
train_set = 2560

# Loss functions (EXACTLY like your original)
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

# Optimizer (same as your original)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if __name__ == "__main__":
    print(f'Learning rate: {learning_rate}, Max_epoch: {Max_epoch}')
    print(f'Model: {Model_name}, Batch size: {batch_size}')
    print(f'Device: {device}')

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    Whole_Loss = []

    with open(log_file, "w") as f:
        f.write(f"Training log for {Model_name}\\n")
        f.write(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Max epochs: {Max_epoch}\\n")
        f.write(f"Training started at: {datetime.datetime.now()}\\n\\n")

        for epoch in range(1, Max_epoch + 1):
            print(f"\\n=== Epoch {epoch}/{Max_epoch} ===")

            # Model training
            model.train()
            train_loader = DataLoader(Dataset, num_workers=8, batch_size=batch_size, shuffle=True)

            # Epoch loss accumulators
            epoch_loss = 0.0
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0

            start_time = time.time()

            # Proper data flow: inputs [bs,256,422] -> model -> outputs [bs,32,256,256]
            for i, (hsi, inputs, labels, mask) in enumerate(train_loader):
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

                # Normalize inputs - exactly like original
                if len(inputs.shape) == 3:  # [bs, H, W]
                    inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                else:  # [bs, C, H, W]
                    inputs_norm = inputs
                    for c in range(inputs.size(1)):
                        inputs_norm[:, c] = (inputs[:, c] - inputs[:, c].min()) / (inputs[:, c].max() - inputs[:, c].min() + 1e-8)

                # Fix mask spatial dimensions: [84, 256, 422] → [84, 256, 256]
                mask_fixed = mask[:, :, :, :256]  # Crop width from 422 to 256

                # MST now handles YOUR mask dimensions
                outputs = model(inputs_norm, mask_fixed)

                # Normalize only labels
                labels = max_min_norm(labels)

                # Calculate losses (same as your original)
                Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
                L1_loss = loss_L1(outputs, labels)
                SSIM_loss = loss_SSIM(outputs, labels)
                loss = Charbonnier_loss + L1_loss + SSIM_loss

                # Accumulate losses
                epoch_loss += loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                clip_grad_norm_(model.parameters(), max_norm=0.2)

                optimizer.step()
                scheduler.step()

                # Log progress (same as your original)
                if i % (50) == 0:
                    log_str = f'{epoch:4d} {i:4d} / {train_set // batch_size:4d} whole_loss = {epoch_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {train_set // batch_size:4d} Charbonnierloss = {epoch_Charbonnier_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {train_set // batch_size:4d} L1_loss = {epoch_L1_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {train_set // batch_size:4d} SSIM_loss = {epoch_SSIM_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'

                    f.write(log_str)
                    f.flush()
                    print(log_str.strip())

            elapsed_time = time.time() - start_time
            avg_loss = epoch_loss / (train_set // batch_size)
            epoch_log_str = f'epoch = {epoch:4d} , loss = {avg_loss:.10f} , time = {elapsed_time:.2f} s\\n'

            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str.strip())

            Whole_Loss.append(avg_loss)

            # Save model checkpoint (same interval as your original)
            if epoch % 50 == 0 or epoch == Max_epoch:
                model_save_path = f"{file_path}parameter/ours/{Model_name}_{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{epoch}_{train_set}_hybridLoss.pth"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model, model_save_path)
                print(f"Model saved at epoch {epoch}: {model_save_path}")

        f.write(f"\\nTraining completed at: {datetime.datetime.now()}\\n")

    # Save final model (same as your original)
    final_model_path = f"{file_path}parameter/ours/{Model_name}/{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{Max_epoch}_{train_set}_hybridLoss.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model, final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Plot loss curve (same as your original)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(Whole_Loss) + 1), Whole_Loss, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{Model_name} Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot_path = f"{file_path}parameter/ours/{Model_name}_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Loss curve saved: {loss_plot_path}")
    print("Training completed successfully!")