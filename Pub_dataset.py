# import torch.utils.data as tud
# import random
# import os
# import torch
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from PIL import Image
# import pickle
# import h5py

# class dataset(tud.Dataset):
#     def __init__(self, HSI_filepath, Label_filepath, is_train = True):
#         super(dataset, self).__init__()
#         self.size = 256
#         self.train_set = 2560 #和train.py中的train_set一致
#         self.is_Train = is_train

#         # 载入HSI数据
#         self.data = dataset.load_mat_files(HSI_filepath)
#         # 载入Label数据
#         self.label = dataset.load_label_files(Label_filepath)

#         # 使用loadmat加载MAT文件
#         mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
#         self.mask = mat_data['mask']
#         self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  # 第三维度复制84次

#     # 载入单张HSI图像（公共数据集）
#     def load_mat_files(base_path):
#         data = []
#         print('loading mat file')
#         with h5py.File(base_path, 'r') as mat:
#             extracted_data = mat['extracted_data'][:]
#             extracted_data = extracted_data.transpose(2, 1, 0)
#             data.append(extracted_data)
#         return data
    
#     # 载入单张Label（公共数据集）
#     def load_label_files(base_path):
#         data = []
#         print('loading label file')
#         with h5py.File(base_path, 'r') as mat:  
#             extracted_data = mat['spectral_indices'][:]
#             extracted_data = extracted_data.transpose(2, 1, 0)
#             data.append(extracted_data)
#         return data

#     def __len__(self):
#         return self.train_set

#     def __getitem__(self, idx):
#         # 获取HSI和Label数据
#         hsi = self.data[0] / 2600
#         target = self.label[0]
        
#         shape1 = np.shape(hsi)
#         # print(shape)
#         cut_index = int(shape1[1] * 0.8)

#         # 如果是训练集，截取左边80%
#         if self.is_Train:
#             hsi = hsi[:, :cut_index, :]
#             target = target[:, :cut_index, :]
#         else:
#             # 如果是测试集，截取右边20%
#             hsi = hsi[:, cut_index:, :]
#             target = target[:, cut_index:, :]
#         # print(np.shape(hsi),np.shape(target))
#         shape = np.shape(hsi)
#         # print(shape)

#         # 对HSI,label.mask进行剪切，使其变成 self.size × self.size
#         # 通过随机数，随机剪切其中的一部分
#         px = random.randint(0, shape[0] - self.size)
#         py = random.randint(0, shape[1] - self.size)
#         hsi = hsi[px:px + self.size:1, py:py + self.size:1, :]
#         target = target[px:px + self.size:1, py:py + self.size:1, :]

#         # 对mask进行剪切，使其变成 self.size × self.size
#         pxm = random.randint(0, 256 - self.size)
#         pym = random.randint(0, 256 - self.size)
#         mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]


#         # 进行翻转
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

#         # 生成测量帧
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

#         #相当于模拟现实世界的不确定性，引入噪声；
#         QE, bit = 0.4, 2048
#         input = np.random.binomial((input * bit / QE).astype(int), QE)
#         input = np.float32(input) / np.float32(bit)
#         input = torch.FloatTensor(input.copy())
#         target = torch.FloatTensor(target.copy()).permute(2,0,1)
#         mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)


#         return input, target


import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import h5py

class dataset(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath, mask_path="/data4/zhuo-file/mask_sim.mat", is_train=True):
        super(dataset, self).__init__()
        self.size = 256
        self.train_set = 2560  # 和训练集保持一致
        self.is_Train = is_train

        # 载入HSI数据（公共数据集：单文件多数据）
        self.data = dataset.load_mat_files(HSI_filepath)
        # 载入Label数据（公共数据集：单文件多数据）
        self.label = dataset.load_label_files(Label_filepath)

        # 加载mask（与MST数据集一致）
        try:
            mat_data = sio.loadmat(mask_path)
            self.mask = mat_data['mask']
        except Exception as e:
            print(f"Warning: Load mask failed! {e}, create random mask")
            self.mask = np.random.choice([0, 1], size=(256, 256), p=[0.5, 0.5])
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  # [256,256,84]

    @staticmethod
    def load_mat_files(base_path):
        """加载公共数据集HSI（h5py格式，适配v7.3+ .mat文件）"""
        data = []
        print(f'Loading HSI mat file: {base_path}')
        with h5py.File(base_path, 'r') as mat:
            # 公共数据集的key是'extracted_data'，转置为[H, W, C]格式（与MST一致）
            extracted_data = mat['extracted_data'][:].transpose(2, 1, 0)  # 调整维度顺序
            data.append(extracted_data)
        return data

    @staticmethod
    def load_label_files(base_path):
        """加载公共数据集标签（h5py格式）"""
        data = []
        print(f'Loading Label mat file: {base_path}')
        with h5py.File(base_path, 'r') as mat:
            # 标签key是'spectral_indices'，转置为[H, W, C]格式
            extracted_data = mat['spectral_indices'][:].transpose(2, 1, 0)
            data.append(extracted_data)
        return data

    def __len__(self):
        return self.train_set

    def __getitem__(self, idx):
        # 读取HSI和标签（公共数据集是单文件，取第0个元素）
        hsi_raw = self.data[0].copy()  # [H, W, C]
        target_raw = self.label[0].copy()  # [H, W, C]

        # 公共数据集分割：训练集取左80%，测试集取右20%（避免数据泄露）
        shape1 = hsi_raw.shape
        cut_index = int(shape1[1] * 0.8)  # 宽度方向分割点
        if self.is_Train:
            hsi_cut = hsi_raw[:, :cut_index, :]
            target_cut = target_raw[:, :cut_index, :]
        else:
            hsi_cut = hsi_raw[:, cut_index:, :]
            target_cut = target_raw[:, cut_index:, :]

        # 随机裁剪到256x256（与MST数据集一致）
        shape = hsi_cut.shape
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        # HSI裁剪：[px:px+256, py:py+256, :]
        hsi = hsi_cut[px:px+self.size, py:py+self.size, :] / 2600  # 归一化到0~1（公共数据集特有）
        # 标签裁剪
        target = target_cut[px:px+self.size, py:py+self.size, :]

        # Mask裁剪（与MST一致）
        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[pxm:pxm+self.size, pym:pym+self.size, :]

        # 训练集数据增强（与MST一致，测试集不增强）
        if self.is_Train:
            # 随机旋转0/90/180/270度
            rotTimes = random.randint(0, 3)
            for _ in range(rotTimes):
                hsi = np.rot90(hsi)
                target = np.rot90(target)
            # 随机垂直翻转
            if random.randint(0, 1):
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()
            # 随机水平翻转
            if random.randint(0, 1):
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        # 生成CASSI测量值（与MST一致的测量模型）
        temp = mask_3d * hsi  # [256,256,84]
        # 移位操作：宽度扩展到256 + 2*(84-1) = 422
        temp_shift = np.zeros((self.size, self.size + (84 - 1)*2, 84))
        mask_3d_shift = np.zeros_like(temp_shift)
        temp_shift[:, :self.size, :] = temp
        mask_3d_shift[:, :self.size, :] = mask_3d
        # 每个通道移位2*t
        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2*t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2*t, axis=1)
        # 求和得到测量值
        meas = np.sum(temp_shift, axis=2)  # [256, 422]
        input_meas = meas / 84 * 0.9  # 归一化测量值

        # 模拟噪声（公共数据集特有，与训练保持一致）
        QE, bit = 0.4, 2048
        input_meas = np.random.binomial((input_meas * bit / QE).astype(int), QE)
        input_meas = np.float32(input_meas) / bit

        # 维度转换：[H, W, C] → [C, H, W]（PyTorch格式）
        # hsi_tensor = torch.FloatTensor(hsi.transpose(2, 0, 1))  # [C, 256, 256]
        hsi_tensor = torch.FloatTensor(hsi.transpose(2, 0, 1).copy())
        # input_tensor = torch.FloatTensor(input_meas)  # [256, 422]（测量值无通道维）
        input_tensor = torch.FloatTensor(input_meas.copy())
        # target_tensor = torch.FloatTensor(target.transpose(2, 0, 1))  # [C, 256, 256]
        target_tensor = torch.FloatTensor(target.transpose(2, 0, 1).copy())
        # mask_tensor = torch.FloatTensor(mask_3d_shift.transpose(2, 0, 1))  # [84, 256, 422]
        mask_tensor = torch.FloatTensor(mask_3d_shift.transpose(2, 0, 1).copy())

        # 关键：返回4个元素（hsi, input, target, mask），与MST数据集完全匹配！
        return hsi_tensor, input_tensor, target_tensor, mask_tensor

# 兼容旧代码的包装器
def Dataset(*args, **kwargs):
    return dataset(*args, **kwargs)