# import torch.utils.data as tud
# import random
# import os
# import torch
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from PIL import Image
# import pickle

# class MST_dataset_corrected(tud.Dataset):
#     """
#     CORRECTED MST dataset that matches your original project exactly:
#     - Returns 2D measurements [256, 422] and spectral indices [32, 256, 256]
#     - Same pipeline as your original dataset
#     - Same data split logic
#     """
#     def __init__(self, HSI_filepath, Label_filepath, mask_path="/data4/zhuo-file/mask_sim.mat"):
#         super(MST_dataset_corrected, self).__init__()
#         self.size = 256
#         self.train_set = 2560
#         self.is_Train = True

#         # Load only necessary files based on train/test mode
#         if self.is_Train:
#             start_num = 1    # Train: load files 1-200
#             end_num = 200
#         else:
#             start_num = 201  # Test: load files 201-255 only
#             end_num = 255

#         # Load HSI data (only the required range)
#         self.data = MST_dataset_corrected.load_mat_files(HSI_filepath, start_num, end_num)
#         # Load spectral indices labels (only the required range)
#         self.label = MST_dataset_corrected.load_label_files(Label_filepath, start_num, end_num)

#         # Load mask for CASSI simulation (exactly like your original)
#         try:
#             mat_data = sio.loadmat(mask_path)
#             self.mask = mat_data['mask']
#         except:
#             print(f"Warning: Could not load mask from {mask_path}, creating random mask")
#             self.mask = np.random.choice([0, 1], size=(256, 256), p=[0.5, 0.5])

#         self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

#     @staticmethod
#     def load_mat_files(base_path, start_num, end_num):
#         """Load HSI data files in specified range"""
#         data = []
#         for num in range(start_num, end_num + 1):
#             if num == 15 or num == 43 or num == 109:
#                 continue
#             print(f'loading mat {num}')
#             filename = os.path.join(base_path, f"{num}.mat")
#             try:
#                 mat = sio.loadmat(filename)
#                 data.append(mat['extracted_data'])
#             except Exception as e:
#                 print(f"Warning: Could not load {filename}: {e}")
#                 continue
#         return data

#     @staticmethod
#     def load_label_files(base_path, start_num, end_num):
#         """Load spectral indices labels in specified range"""
#         data = []
#         for num in range(start_num, end_num + 1):
#             if num == 15 or num == 43 or num == 109:
#                 continue
#             print(f'loading label {num}')
#             filename = os.path.join(base_path, f"{num}_indices.mat")
#             try:
#                 mat = sio.loadmat(filename)
#                 data.append(mat['spectral_indices'])
#             except Exception as e:
#                 print(f"Warning: Could not load {filename}: {e}")
#                 continue
#         return data

#     def __len__(self):
#         return self.train_set

#     def __getitem__(self, idx):
#         # Data split: use random file from loaded range
#         # Training loads files 1-200 → data indices 0-199
#         # Testing loads files 201-255 → data indices 0-54
#         index1 = random.randint(0, len(self.data) - 1)
#         hsi = self.data[index1]
#         target = self.label[index1]

#         shape = np.shape(hsi)

#         # Random cropping (exactly like your original)
#         px = random.randint(0, shape[0] - self.size)
#         py = random.randint(0, shape[1] - self.size)
#         hsi = hsi[px:px + self.size:1, py:py + self.size:1, :]
#         target = target[px:px + self.size:1, py:py + self.size:1, :]

#         # Crop mask (exactly like your original)
#         pxm = random.randint(0, 256 - self.size)
#         pym = random.randint(0, 256 - self.size)
#         mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

#         # Data augmentation (exactly like your original)
#         if self.is_Train:
#             rotTimes = random.randint(0, 3)
#             vFlip = random.randint(0, 1)
#             hFlip = random.randint(0, 1)

#             # Random rotation
#             for j in range(rotTimes):
#                 hsi = np.rot90(hsi)
#                 target = np.rot90(target)

#             # Random vertical Flip
#             for j in range(vFlip):
#                 hsi = hsi[:, ::-1, :].copy()
#                 target = target[:, ::-1, :].copy()

#             # Random horizontal Flip
#             for j in range(hFlip):
#                 hsi = hsi[::-1, :, :].copy()
#                 target = target[::-1, :, :].copy()

#         # Generate CASSI measurement (EXACTLY like your original)
#         temp = mask_3d * hsi
#         temp_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
#         temp_shift[:, 0:self.size, :] = temp
#         mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
#         mask_3d_shift[:, 0:self.size, :] = mask_3d

#         for t in range(84):
#             temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
#             mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
#         meas = np.sum(temp_shift, axis=2)

#         input = meas / 84 * 0.9

#         # Convert to tensors (EXACTLY like your original)
#         input = torch.FloatTensor(input.copy())
#         hsi = torch.FloatTensor(hsi.astype(np.float32).copy()).permute(2,0,1)  # Convert uint16 to float32
#         target = torch.FloatTensor(target.astype(np.float32).copy()).permute(2,0,1)  # Convert uint16 to float32

#         # Return 4 values: hsi, inputs, labels, mask for MST
#         mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)  # [84, 256, 422]
#         return hsi, input, target, mask_3d_shift

# # Wrapper for backward compatibility
# def dataset(*args, **kwargs):
#     """Wrapper function for backward compatibility"""
#     return MST_dataset_corrected(*args, **kwargs)


import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class MST_dataset_corrected(tud.Dataset):
    """
    修复后的MST数据集类：
    - 允许在初始化时指定训练/测试模式
    - 确保测试模式下从201开始加载数据
    """
    def __init__(self, HSI_filepath, Label_filepath, mask_path="/data4/zhuo-file/mask_sim.mat", is_train=True):
        super(MST_dataset_corrected, self).__init__()
        self.size = 256
        self.train_set = 2560
        self.is_Train = is_train  # 从参数接收模式，而不是硬编码为True

        # 根据初始化时指定的模式加载对应范围的数据
        if self.is_Train:
            start_num = 1    # 训练模式：加载1-200
            end_num = 200
        else:
            start_num = 201  # 测试模式：加载201-255
            end_num = 255

        # 加载数据（此时会根据上面的start_num和end_num正确加载）
        self.data = MST_dataset_corrected.load_mat_files(HSI_filepath, start_num, end_num)
        self.label = MST_dataset_corrected.load_label_files(Label_filepath, start_num, end_num)

        # 加载掩码（保持不变）
        try:
            mat_data = sio.loadmat(mask_path)
            self.mask = mat_data['mask']
        except:
            print(f"Warning: Could not load mask from {mask_path}, creating random mask")
            self.mask = np.random.choice([0, 1], size=(256, 256), p=[0.5, 0.5])

        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

    # 以下方法保持不变...
    @staticmethod
    def load_mat_files(base_path, start_num, end_num):
        data = []
        for num in range(start_num, end_num + 1):
            if num == 15 or num == 43 or num == 109:
                continue
            print(f'loading mat {num}')
            filename = os.path.join(base_path, f"{num}.mat")
            try:
                mat = sio.loadmat(filename)
                data.append(mat['extracted_data'])
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
        return data

    @staticmethod
    def load_label_files(base_path, start_num, end_num):
        data = []
        for num in range(start_num, end_num + 1):
            if num == 15 or num == 43 or num == 109:
                continue
            print(f'loading label {num}')
            filename = os.path.join(base_path, f"{num}_indices.mat")
            try:
                mat = sio.loadmat(filename)
                data.append(mat['spectral_indices'])
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
        return data

    def __len__(self):
        return self.train_set

    def __getitem__(self, idx):
        # 保持原有实现...
        index1 = random.randint(0, len(self.data) - 1)
        hsi = self.data[index1]
        target = self.label[index1]

        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        hsi = hsi[px:px + self.size:1, py:py + self.size:1, :]
        target = target[px:px + self.size:1, py:py + self.size:1, :]

        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        if self.is_Train:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            for j in range(rotTimes):
                hsi = np.rot90(hsi)
                target = np.rot90(target)

            for j in range(vFlip):
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()

            for j in range(hFlip):
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        temp = mask_3d * hsi
        temp_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        temp_shift[:, 0:self.size, :] = temp
        mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:self.size, :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)

        input = meas / 84 * 0.9

        input = torch.FloatTensor(input.copy())
        hsi = torch.FloatTensor(hsi.astype(np.float32).copy()).permute(2,0,1)
        target = torch.FloatTensor(target.astype(np.float32).copy()).permute(2,0,1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)
        
        return hsi, input, target, mask_3d_shift

def dataset(*args, **kwargs):
    return MST_dataset_corrected(*args, **kwargs)
