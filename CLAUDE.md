# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research codebase focused on hyperspectral image (HSI) processing and deep learning models. The project implements various neural network architectures including U-Net variations, attention mechanisms (ATTU), and Multi-Scale Transformer (MST) models for hyperspectral image analysis.

## Key Architecture Components

### Model Directory (`Model/`)
- **ATTU.py**: Attention-based U-Net with Transformer components, including Multi-Scale Attention blocks (MSABC)
- **UNet.py** & **UNet_KAN.py**: Standard and KAN-enhanced U-Net implementations
- **TRM_Block.py** & **TRM_Block_with_CNN.py**: Transformer block implementations with and without CNN integration
- **KAN.py**: Kolmogorov-Arnold Network implementation
- **MST/**: Multi-Scale Transformer models directory
  - **MST.py**: Core Multi-Scale Transformer implementation
  - **MST_GPT.py**: GPT-enhanced MST variant
  - **MSAB.py**: Multi-Scale Attention Block component

### Training Infrastructure (`train/`)
- **train.py**: Main training script for standard models
- **train_mst.py**: Specialized training for MST models
- **dataset.py**: Dataset loading and preprocessing
- **loss.py**: Custom loss functions including CharbonnierLoss and SSIM
- **Norm.py**: Normalization utilities for hyperspectral data

### Testing Infrastructure (`test/`)
- **test.py**: Main testing script
- **test_mst.py**: MST model testing
- **dataset.py**, **lz_dataset.py**, **Pub_dataset.py**: Various dataset implementations for different test scenarios

## Data Processing

The codebase processes hyperspectral image data with specific characteristics:
- Input data: 84-layer hyperspectral images stored as .mat files
- Data filtering: Images with layer maximum values > 16000 are automatically skipped
- Normalization: Custom min-max normalization for hyperspectral data ranges

## Common Development Commands

### Training Models
```bash
# Standard model training
python train/train.py

# MST model training
python train/train_mst.py
```

### Testing Models
```bash
# Standard model testing
python test/test.py

# MST model testing
python test/test_mst.py
```

### Data Analysis
```bash
# Analyze hyperspectral data ranges
python Range_view.py
```

## GPU Configuration

Models are configured to use CUDA device 3 by default:
```python
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
```

## Key Hyperparameters

- Learning rate: 0.0004
- Batch size: 8 (training), 1 (testing)
- Training epochs: 500
- Training set size: 2560 samples
- Test set size: 1200 samples

## Loss Functions

The codebase implements hybrid loss combining:
- L1 Loss
- Charbonnier Loss
- SSIM (Structural Similarity Index)

## File Structure Notes

- Model weights are saved in `parameter/ours/` directory
- Training logs use descriptive naming: `{Model_name}_lr{lr}_bs{batch_size}_Epoch{epochs}_{train_set}_hybridLoss`
- Hard-coded paths point to `/data4/zhuo-file/` - update these for your environment