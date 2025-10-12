import os
import scipy.io as sio
import numpy as np
from PIL import Image
import argparse

"""
Visualization script for MST_Corrected spectral indices output.
Reads .mat files saved by mst_test_corrected.py and generates individual
grayscale PNG images for each of the 32 spectral indices.
"""

# Configuration
RESULTS_DIR = "./MST_Corrected_test_results_com/"
# RESULTS_DIR = "./Pub_MST_Corrected_test_results/"
NUM_SAMPLES = 5  # Number of test samples to visualize
NUM_CHANNELS = 32  # Number of spectral indices

def visualize_sample(mat_file_path, sample_idx):
    """
    Visualize a single sample by saving all 32 channels as separate PNG images.

    Args:
        mat_file_path: Path to .mat file containing 'output' and 'label'
        sample_idx: Sample index for naming output directories
    """
    print(f"\n{'='*60}")
    print(f"Processing: {mat_file_path}")
    print(f"{'='*60}")

    # Check if file exists
    if not os.path.exists(mat_file_path):
        print(f"ERROR: File not found: {mat_file_path}")
        return

    # Load .mat file
    try:
        data = sio.loadmat(mat_file_path)
        output = data['output']  # Model prediction [32, 256, 256]
        label = data['label']    # Ground truth [32, 256, 256]
        print(f"Loaded data shapes - Output: {output.shape}, Label: {label.shape}")
    except Exception as e:
        print(f"ERROR loading .mat file: {e}")
        return

    # Create output directories
    output_dir = os.path.join(RESULTS_DIR, f"sample_{sample_idx}_output_channels")
    label_dir = os.path.join(RESULTS_DIR, f"sample_{sample_idx}_label_channels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Process both output and label
    for data_type, data_array, save_dir in [
        ("Output", output, output_dir),
        ("Label", label, label_dir)
    ]:
        print(f"\n--- Processing {data_type} ---")
        C, H, W = data_array.shape

        # Save each channel as grayscale PNG
        for i in range(C):
            # Extract channel data
            channel_data = data_array[i, :, :]

            # Get min/max for normalization
            channel_min = channel_data.min()
            channel_max = channel_data.max()

            # Check if all values are identical
            if channel_max == channel_min:
                print(f"  Channel {i+1:2d}: All identical values ({channel_min:.6f}) - Saving as black image")
                channel_norm = np.zeros_like(channel_data)
            else:
                # Normalize to [0, 1]
                channel_norm = (channel_data - channel_min) / (channel_max - channel_min)
                print(f"  Channel {i+1:2d}: min={channel_min:.6f}, max={channel_max:.6f}")

            # Convert to [0, 255] uint8
            channel_img = (channel_norm * 255).astype(np.uint8)

            # Save as grayscale PNG
            img = Image.fromarray(channel_img, mode='L')
            save_path = os.path.join(save_dir, f"channel_{i+1:02d}.png")
            img.save(save_path)

        print(f"✓ Saved {C} channel images to: {save_dir}")

def create_comparison_grid(sample_idx):
    """
    Create a single image showing all 32 channels in a grid for easy comparison.

    Args:
        sample_idx: Sample index to create grid for
    """
    mat_file_path = os.path.join(RESULTS_DIR, f"sample_{sample_idx}_mst_corrected.mat")

    if not os.path.exists(mat_file_path):
        print(f"Skipping grid for sample {sample_idx} - file not found")
        return

    # Load data
    data = sio.loadmat(mat_file_path)
    output = data['output']
    label = data['label']

    import matplotlib.pyplot as plt

    # Create figure with 2 rows (output, label) and 32/4 = 8 columns (showing every 4th channel)
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f'Sample {sample_idx}: All Spectral Indices (Every 4th Channel)', fontsize=16, y=0.98)

    # Plot every 4th channel (0, 4, 8, ..., 28)
    for col_idx, ch_idx in enumerate(range(0, 32, 4)):
        # Output row
        ax_out = axes[0, col_idx]
        channel_out = output[ch_idx, :, :]
        if channel_out.max() > channel_out.min():
            channel_out_norm = (channel_out - channel_out.min()) / (channel_out.max() - channel_out.min())
        else:
            channel_out_norm = np.zeros_like(channel_out)
        ax_out.imshow(channel_out_norm, cmap='gray', vmin=0, vmax=1)
        ax_out.set_title(f'Ch {ch_idx+1} (Out)', fontsize=10)
        ax_out.axis('off')

        # Label row
        ax_label = axes[1, col_idx]
        channel_label = label[ch_idx, :, :]
        if channel_label.max() > channel_label.min():
            channel_label_norm = (channel_label - channel_label.min()) / (channel_label.max() - channel_label.min())
        else:
            channel_label_norm = np.zeros_like(channel_label)
        ax_label.imshow(channel_label_norm, cmap='gray', vmin=0, vmax=1)
        ax_label.set_title(f'Ch {ch_idx+1} (GT)', fontsize=10)
        ax_label.axis('off')

    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Model Output', transform=axes[0, 0].transAxes,
                   fontsize=12, va='center', ha='right', rotation=90, weight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Ground Truth', transform=axes[1, 0].transAxes,
                   fontsize=12, va='center', ha='right', rotation=90, weight='bold')

    plt.tight_layout()

    # Save grid image
    grid_path = os.path.join(RESULTS_DIR, f"sample_{sample_idx}_all_channels_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison grid to: {grid_path}")

def main():
    """Main visualization pipeline"""
    print("="*60)
    print("MST_Corrected Spectral Indices Visualization")
    print("="*60)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Number of channels per sample: {NUM_CHANNELS}")

    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"\nERROR: Results directory not found: {RESULTS_DIR}")
        print("Please run mst_test_corrected.py first to generate test results.")
        return

    # Process each sample
    for sample_idx in range(NUM_SAMPLES):
        mat_file = os.path.join(RESULTS_DIR, f"sample_{sample_idx}_mst_corrected.mat")
        visualize_sample(mat_file, sample_idx)

        # Also create comparison grid
        try:
            create_comparison_grid(sample_idx)
        except Exception as e:
            print(f"Warning: Could not create comparison grid for sample {sample_idx}: {e}")

    print("\n" + "="*60)
    print("✓ Visualization completed!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {RESULTS_DIR}")
    print(f"    ├─ sample_0_output_channels/  (32 PNG files)")
    print(f"    ├─ sample_0_label_channels/   (32 PNG files)")
    print(f"    ├─ sample_0_all_channels_grid.png")
    print(f"    ├─ sample_1_output_channels/")
    print(f"    └─ ... (and so on for all {NUM_SAMPLES} samples)")
    print(f"\nTotal images generated:")
    print(f"  - Individual channel images: {NUM_SAMPLES} samples × {NUM_CHANNELS} channels × 2 (output+label) = {NUM_SAMPLES * NUM_CHANNELS * 2} PNG files")
    print(f"  - Comparison grids: {NUM_SAMPLES} PNG files")
    print(f"  - Grand total: {NUM_SAMPLES * NUM_CHANNELS * 2 + NUM_SAMPLES} images")

if __name__ == "__main__":
    main()
