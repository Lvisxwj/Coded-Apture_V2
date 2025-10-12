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
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Import loss functions and normalization (local copies)
from loss import CharbonnierLoss, SSIM
from Norm import max_min_norm

# Data paths (same as your original)
hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/cor_r_label/"

# Load dataset and set to test mode
Dataset = MST_dataset_corrected(hsi_filename, label_filename, is_train=False)
# Dataset.is_Train = False  # Set to test mode

# GPU setup (same as your original)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load trained model
# model_path = r"/home/graduate/lizhuo/xwj/parameter/ours/MST_Corrected/MST_Corrected_lr4-4_bs8_Epoch500_2560_hybridLoss.pth"
# model_path = r"/home/graduate/lizhuo/xwj/parameter/ours/MST_Corrected/MST_Corrected_lr1e-04_bs4_Epoch500_2560_hybridLoss.pth"

model_path = r"/home/graduate/lizhuo/xwj/parameter/ours/MST_Corrected_MST_Corrected_lr1e-04_bs4_Epoch500_2560_hybridLoss.pth"

try:
    model = torch.load(model_path, map_location=device)
    print(f"Loaded model from: {model_path}")
except:
    print(f"Could not load model from {model_path}, initializing new model")
    model = MST_Corrected(dim=64, stage=3, num_blocks=[2,2,2])

model.to(device)

# Test parameters (same as your original)
batch_size = 1
Max_epoch = 1
test_set = 1200

# Loss functions (same as your original)
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

if __name__ == "__main__":
    # log_file = r"./MST_Corrected_lr1e-04_bs4_Epoch500_test_log.txt"
    log_file = r"./MST_Corrected_lr1e-04_bs4_Epoch500_Com_test_log.txt"

    print(f'Testing MST_Corrected model on {device}')
    print(f'Test set size: {test_set}')

    Whole_Loss = []

    with open(log_file, "w") as f:
        f.write(f"MST_Corrected Test Results\\n")
        f.write(f"Model: {model_path}\\n")
        f.write(f"Test started at: {datetime.datetime.now()}\\n\\n")

        for epoch in range(1, Max_epoch + 1):
            print(f"\\n=== Test Epoch {epoch}/{Max_epoch} ===")

            # Model evaluation
            model.eval()
            test_loader = DataLoader(Dataset, num_workers=8, batch_size=batch_size, shuffle=False)

            # Initialize loss accumulators
            epoch_loss = 0.0
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0

            all_outputs = []
            all_labels = []

            start_time = time.time()

            # FIXED: Correct unpacking - 4 items from dataset (hsi, input, target, mask)
            for i, (hsi, inputs, labels, mask) in enumerate(test_loader):
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

                # Normalize inputs - exactly like training
                if len(inputs.shape) == 3:  # [bs, H, W]
                    inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                else:  # [bs, C, H, W]
                    inputs_norm = inputs
                    for c in range(inputs.size(1)):
                        inputs_norm[:, c] = (inputs[:, c] - inputs[:, c].min()) / (inputs[:, c].max() - inputs[:, c].min() + 1e-8)

                # Fix mask spatial dimensions: [84, 256, 422] → [84, 256, 256] - exactly like training
                mask_fixed = mask[:, :, :, :256]  # Crop width from 422 to 256

                with torch.no_grad():
                    outputs = model(inputs_norm, mask_fixed)

                print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Label shape: {labels.shape}")

                # FIXED: Normalize BOTH outputs and labels (fix the bug!)
                outputs = max_min_norm(outputs)  # Normalize model outputs
                labels = max_min_norm(labels)    # Normalize ground truth

                # Calculate losses (same as your original)
                Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
                L1_loss = loss_L1(outputs, labels)
                SSIM_loss = loss_SSIM(outputs, labels)
                loss = Charbonnier_loss + L1_loss + SSIM_loss

                epoch_loss += loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()

                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Log results (same as your original)
                if i % (100) == 0:
                    log_str = f'{epoch:4d} {i:4d} / {test_set // batch_size:4d} whole_loss = {epoch_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {test_set // batch_size:4d} Charbonnierloss = {epoch_Charbonnier_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {test_set // batch_size:4d} L1_loss = {epoch_L1_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'
                    log_str += f'{epoch:4d} {i:4d} / {test_set // batch_size:4d} SSIM_loss = {epoch_SSIM_loss / (i + 1):.10f} time = {datetime.datetime.now()}\\n'

                    f.write(log_str)
                    f.flush()
                    print(log_str.strip())

            elapsed_time = time.time() - start_time
            avg_loss = epoch_loss / (test_set // batch_size)
            epoch_log_str = f'epoch = {epoch:4d} , loss = {avg_loss:.10f} , time = {elapsed_time:.2f} s\\n'

            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str.strip())

            Whole_Loss.append(avg_loss)

            # Save some sample results for visualization
            results_dir = "./MST_Corrected_test_results_com/"
            os.makedirs(results_dir, exist_ok=True)

            # Save first few samples as .mat files
            for idx in range(min(5, len(all_outputs))):
                output_sample = all_outputs[idx][0]  # First item in batch [32, 256, 256]
                label_sample = all_labels[idx][0]    # First item in batch [32, 256, 256]

                sio.savemat(
                    os.path.join(results_dir, f'sample_{idx}_mst_corrected.mat'),
                    {
                        'output': output_sample,
                        'label': label_sample,
                        'method': 'MST_Corrected'
                    }
                )

                # Create visualization using some spectral indices
                if output_sample.shape[0] >= 3:
                    # Use first 3 channels as RGB
                    rgb_output = np.stack([
                        output_sample[0],   # Red
                        output_sample[1],   # Green
                        output_sample[2]    # Blue
                    ], axis=2)

                    rgb_label = np.stack([
                        label_sample[0],
                        label_sample[1],
                        label_sample[2]
                    ], axis=2)

                    # Normalize for visualization
                    rgb_output = (rgb_output - rgb_output.min()) / (rgb_output.max() - rgb_output.min() + 1e-8)
                    rgb_label = (rgb_label - rgb_label.min()) / (rgb_label.max() - rgb_label.min() + 1e-8)

                    # Save comparison plot
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(rgb_output)
                    plt.title('MST_Corrected Output')
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(rgb_label)
                    plt.title('Ground Truth')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f'sample_{idx}_comparison.png'),
                              dpi=300, bbox_inches='tight')
                    plt.close()

        f.write(f"\\nTesting completed at: {datetime.datetime.now()}\\n")

    print(f"Test results saved to: {results_dir}")
    print(f"Test log saved to: {log_file}")
    print("Testing completed successfully!")

    # Final summary
    print(f"\\n=== Final Test Results ===")
    print(f"Average Loss: {Whole_Loss[0]:.10f}")
    print(f"Test completed for MST_Corrected model")
    
#灰度图