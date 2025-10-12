import os
import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
公共数据集MST_Corrected光谱指数可视化脚本
适配文件名：pub_sample_0_mst_corrected.mat
"""

# 配置（与测试代码生成的文件路径和命名严格匹配）
RESULTS_DIR = "./Pub_MST_Corrected_test_results/"  # 结果目录
NUM_SAMPLES = 5  # 可视化样本数量（与测试代码保存的样本数一致）
NUM_CHANNELS = 32  # 光谱指数通道数

def visualize_sample(mat_file_path, sample_idx):
    """可视化单个样本的所有通道"""
    print(f"\n{'='*60}")
    print(f"处理文件: {mat_file_path}")
    print(f"{'='*60}")

    if not os.path.exists(mat_file_path):
        print(f"错误: 文件不存在 - {mat_file_path}")
        return

    try:
        data = sio.loadmat(mat_file_path)
        output = data['output']  # 形状 [32, 256, 256]
        label = data['label']    # 形状 [32, 256, 256]
        print(f"数据加载成功 - 输出: {output.shape}, 标签: {label.shape}")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 创建保存目录
    output_dir = os.path.join(RESULTS_DIR, f"pub_sample_{sample_idx}_output_channels")
    label_dir = os.path.join(RESULTS_DIR, f"pub_sample_{sample_idx}_label_channels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 处理输出和标签的每个通道
    for data_type, data_array, save_dir in [
        ("模型输出", output, output_dir),
        ("真实标签", label, label_dir)
    ]:
        print(f"\n--- 处理{data_type} ---")
        C, H, W = data_array.shape

        for i in range(C):
            channel_data = data_array[i, :, :]
            channel_min, channel_max = channel_data.min(), channel_data.max()

            # 归一化到[0, 255]
            if channel_max == channel_min:
                print(f"  通道 {i+1:2d}: 所有值相同 ({channel_min:.6f})")
                channel_norm = np.zeros_like(channel_data)
            else:
                channel_norm = (channel_data - channel_min) / (channel_max - channel_min)
                print(f"  通道 {i+1:2d}: 范围 [{channel_min:.6f}, {channel_max:.6f}]")

            # 保存为灰度图
            channel_img = (channel_norm * 255).astype(np.uint8)
            img = Image.fromarray(channel_img, mode='L')
            save_path = os.path.join(save_dir, f"channel_{i+1:02d}.png")
            img.save(save_path)

        print(f"✓ 已保存 {C} 个通道到 {save_dir}")

def create_comparison_grid(sample_idx):
    """创建通道对比网格图"""
    # 关键修正：匹配实际文件名 pub_sample_xxx.mat
    mat_file_path = os.path.join(RESULTS_DIR, f"pub_sample_{sample_idx}_mst_corrected.mat")
    
    if not os.path.exists(mat_file_path):
        print(f"跳过网格图 - 文件未找到: {mat_file_path}")
        return

    data = sio.loadmat(mat_file_path)
    output = data['output']
    label = data['label']

    # 创建2行8列的网格（显示每4个通道）
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f'样本 {sample_idx} 光谱指数对比（每4个通道）', fontsize=16)

    for col_idx, ch_idx in enumerate(range(0, 32, 4)):
        # 模型输出行
        ax_out = axes[0, col_idx]
        out_norm = (output[ch_idx] - output[ch_idx].min()) / (output[ch_idx].max() - output[ch_idx].min() + 1e-8)
        ax_out.imshow(out_norm, cmap='gray')
        ax_out.set_title(f'通道 {ch_idx+1} (输出)', fontsize=10)
        ax_out.axis('off')

        # 真实标签行
        ax_label = axes[1, col_idx]
        label_norm = (label[ch_idx] - label[ch_idx].min()) / (label[ch_idx].max() - label[ch_idx].min() + 1e-8)
        ax_label.imshow(label_norm, cmap='gray')
        ax_label.set_title(f'通道 {ch_idx+1} (标签)', fontsize=10)
        ax_label.axis('off')

    # 添加行标题
    axes[0, 0].text(-0.2, 0.5, '模型输出', rotation=90, va='center', transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.2, 0.5, '真实标签', rotation=90, va='center', transform=axes[1, 0].transAxes)

    plt.tight_layout()
    grid_path = os.path.join(RESULTS_DIR, f"pub_sample_{sample_idx}_comparison_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 网格对比图已保存到 {grid_path}")

def main():
    print("="*60)
    print("公共数据集MST结果可视化工具")
    print("="*60)
    print(f"结果目录: {RESULTS_DIR}")
    print(f"目标文件名格式: pub_sample_<index>_mst_corrected.mat")

    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 结果目录不存在 - {RESULTS_DIR}")
        return

    # 遍历所有样本
    for sample_idx in range(NUM_SAMPLES):
        # 构造带pub_前缀的文件名
        mat_file = os.path.join(RESULTS_DIR, f"pub_sample_{sample_idx}_mst_corrected.mat")
        visualize_sample(mat_file, sample_idx)
        create_comparison_grid(sample_idx)

    print("\n" + "="*60)
    print("可视化完成!")
    print(f"输出位置: {RESULTS_DIR}")
    print(f"每个样本包含 {NUM_CHANNELS} 个输出通道和 {NUM_CHANNELS} 个标签通道")

if __name__ == "__main__":
    main()
