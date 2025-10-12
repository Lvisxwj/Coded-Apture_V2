# MST Module Architecture

This document describes the MST (Multi-Scale Transformer) integration for your hyperspectral research project.

## Files Structure and Usage

### 🎯 **Essential Files (KEEP AND USE)**

#### **Core Model Files:**
1. **`MST_corrected.py`** - Main MST model adapted for your pipeline
   - **Function**: MST architecture with attention mechanisms
   - **Input**: 2D CASSI measurement `[bs, 256, 422]`
   - **Output**: 32 spectral indices `[bs, 32, 256, 256]`
   - **Usage**: Direct replacement for your UNet models

2. **`mst_dataset_corrected.py`** - Dataset loader matching your original
   - **Function**: Loads HSI data and generates CASSI measurements
   - **Returns**: `(input, target)` - exactly like your original dataset
   - **Input**: 2D measurement `[256, 422]`
   - **Target**: 32 spectral indices `[32, 256, 256]`

3. **`mst_train_corrected.py`** - Training script with fixed normalization
   - **Function**: Trains MST model with corrected normalization
   - **Key Fix**: Normalizes BOTH outputs and labels (solves the bug!)
   - **Usage**: `python mst_train_corrected.py`

4. **`mst_test_corrected.py`** - Testing script with corrected normalization
   - **Function**: Evaluates trained MST model
   - **Usage**: `python mst_test_corrected.py`

#### **Dependency Files:**
5. **`loss.py`** - Loss functions (copied from `train/loss.py`)
   - **Function**: CharbonnierLoss, SSIM, PSNR functions
   - **Why Local Copy**: Ensures consistent imports

6. **`Norm.py`** - Normalization functions (copied from `train/Norm.py`)
   - **Function**: `max_min_norm()` and other normalization utilities
   - **Why Local Copy**: Ensures consistent imports

7. **`__init__.py`** - Module initialization with correct imports
   - **Function**: Makes MST module importable
   - **Imports**: All essential components

### 🗑️ **Obsolete Files (DELETE)**

- `mst_dataset_adapted.py` - Wrong pipeline (HSI→HSI)
- `mst_train_adapted.py` - Wrong pipeline
- `mst_test_adapted.py` - Wrong pipeline
- `mst_dataset_configurable.py` - Wrong pipeline + unnecessary
- `mst_norm.py` - Redundant (use local `Norm.py`)
- `mst_loss.py` - Redundant (use local `loss.py`)
- `README_MST_USAGE.md` - Outdated information

### 📁 **Original MST Files (REFERENCE ONLY)**
- `MST.py` - Original MST implementation
- `MSAB.py` - Original MST components
- `utils.py` - Original utilities

## Tuning Parameters and Configuration

### 🔧 **Data Split Control**

**Location**: `mst_dataset_corrected.py:64-72`

```python
def __getitem__(self, idx):
    if self.is_Train:
        index1 = random.randint(0, 200)    # TRAIN: files 0-200
        hsi = self.data[index1]
        target = self.label[index1]
    else:
        index1 = random.randint(200, min(250, len(self.data)-1))  # TEST: files 200-250
```

**How to Modify:**
```python
# For fewer training examples (e.g., 50 files):
index1 = random.randint(0, 50)     # Train: files 0-50

# For different test split (e.g., files 180-210):
index1 = random.randint(180, 210)  # Test: files 180-210

# For specific file selection:
train_files = [0, 10, 20, 30, 40]  # Specific files
index1 = random.choice(train_files)
```

### ⚙️ **Training Hyperparameters**

**Location**: `mst_train_corrected.py:32-42`

```python
# Hyperparameters (same as your original)
learning_rate = 0.0004        # ← ADJUST HERE
milestones = [200, 400]       # ← ADJUST HERE
gamma = 0.5                   # ← ADJUST HERE
batch_size = 8                # ← ADJUST HERE
Max_epoch = 500               # ← ADJUST HERE
train_set = 2560              # ← ADJUST HERE
```

**Quick Experiment Settings:**
```python
# For quick testing:
learning_rate = 0.001         # Higher LR for faster convergence
batch_size = 4                # Smaller batch if GPU memory limited
Max_epoch = 100               # Fewer epochs for quick test
train_set = 500               # Fewer samples per epoch

# For fewer data:
# Also modify data split in dataset file to use fewer files
```

### 🎛️ **Model Architecture Parameters**

**Location**: `mst_train_corrected.py:28` and `MST_corrected.py:214`

```python
# Model initialization
model = MST_Corrected(
    dim=64,                   # ← ADJUST: Feature dimension (32, 64, 128)
    stage=3,                  # ← ADJUST: Number of encoder/decoder stages (2, 3, 4)
    num_blocks=[2,2,2]        # ← ADJUST: Attention blocks per stage
)
```

**Parameter Effects:**
- **`dim`**: Higher = more features, more memory, potentially better quality
- **`stage`**: More stages = deeper network, more parameters
- **`num_blocks`**: More blocks = more attention, more computation

### 📊 **Data Size Control**

**Location**: `mst_dataset_corrected.py:17`

```python
nums = 210    # ← ADJUST: Number of files to load (default 210)
```

**For Experiments:**
```python
nums = 50     # Load only 50 files for quick experiments
nums = 100    # Load 100 files for medium experiments
nums = 210    # Load all files for full experiments
```

## How to Make It Run

### 🚀 **Step 1: Basic Setup**
```bash
cd Model/MST
```

### 🚀 **Step 2: Quick Test (Small Scale)**

**Modify for quick test:**
```python
# In mst_dataset_corrected.py:
nums = 20                     # Load only 20 files
index1 = random.randint(0, 10)  # Train: 0-10, Test: 10-20

# In mst_train_corrected.py:
learning_rate = 0.001
batch_size = 2
Max_epoch = 10
train_set = 100
```

**Run:**
```bash
python mst_train_corrected.py
```

### 🚀 **Step 3: Full Scale Training**

**Restore original settings:**
```python
# In mst_dataset_corrected.py:
nums = 210
index1 = random.randint(0, 200)  # Original split

# In mst_train_corrected.py:
learning_rate = 0.0004
batch_size = 8
Max_epoch = 500
train_set = 2560
```

**Run:**
```bash
python mst_train_corrected.py
```

### 🚀 **Step 4: Testing**
```bash
python mst_test_corrected.py
```

### 🚀 **Step 5: Different Learning Rate Experiments**

**Quick Parameter Sweep:**
```python
# Try different learning rates:
for lr in [0.0001, 0.0004, 0.001, 0.002]:
    learning_rate = lr
    # Run training and save with different model names
```

## Data Paths Configuration

**Update these paths in both train and test files:**

```python
# In mst_train_corrected.py and mst_test_corrected.py:
hsi_filename = r"/your/path/to/5nm_input/"        # ← UPDATE
label_filename = r"/your/path/to/cor_r_label/"    # ← UPDATE
mask_path = "/your/path/to/mask_sim.mat"          # ← UPDATE (in dataset)
```

## Expected Performance

MST should provide **better reconstruction quality** than your original UNet models due to:
- Multi-scale attention mechanisms
- Better feature extraction
- Improved spatial-spectral modeling

**If performance is worse**, check:
1. Normalization is applied correctly
2. Learning rate isn't too high/low
3. Sufficient training epochs
4. Data paths are correct

## File Import Summary

```python
# Now you can import everything cleanly:
from Model.MST import MST_Corrected, dataset, CharbonnierLoss, SSIM, max_min_norm

# Or use directly:
import Model.MST.mst_train_corrected  # Runs training
import Model.MST.mst_test_corrected   # Runs testing
```

This setup gives you MST-main's architecture applied to your exact research pipeline with full control over data splits, hyperparameters, and model configurations.