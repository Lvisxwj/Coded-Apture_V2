from torch.autograd import Variable
import sys
import os
# 1. 统一结果目录：创建 Pub_train_result，所有输出放这里
result_dir = "./Pub_train_result"
os.makedirs(result_dir, exist_ok=True)
sys.path.append("/data4/zhuo-file/fyc_file/ATTU_Test/")

import torch
from torch.utils.data import DataLoader
from MST_corrected import MST_Corrected
from Pub_dataset import dataset  # 确保 Pub_dataset 与 MST 数据集逻辑一致
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from loss import CharbonnierLoss, SSIM  # 明确导入损失函数
from torch.nn.utils import clip_grad_norm_
from Norm import XA_max_min_norm  # 公共数据集专用归一化函数


# 2. 数据路径与加载：确保与 pub_test 一致
hsi_filename = r"/data4/zhuo-file/extracted_data/Public_Data/XiongAn_jg5.mat"
label_filename = r"/data4/zhuo-file/extracted_data/Public_Data/new_XiongAn_indices.mat"
# 加载训练集：明确传入 is_train=True（关键！只加载训练数据）
Dataset = dataset(hsi_filename, label_filename, is_train=True)


# 3. 模型初始化：与 MST 训练完全一致
model = MST_Corrected(dim=64, stage=3, num_blocks=[2,2,2])
Model_name = "MST_Corrected_XiongAn"
file_path = "/home/graduate/lizhuo/xwj/"

# 统一日志路径：放到结果目录下
log_file = os.path.join(result_dir, f"training_log_{Model_name}_lr1e-04_bs4_Epoch500_2560_hybridLoss_pub.txt")


# 4. GPU 配置
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)


# 5. 超参数：与 MST 训练保持一致
learning_rate = 0.0001
milestones = [200, 400]
gamma = 0.5
batch_size = 4
Max_epoch = 500
train_set = 2560  # 需与 Pub_dataset 的训练集大小一致


# 6. 损失函数：与 MST 训练完全一致
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()


# 7. 优化器：与 MST 训练完全一致
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


if __name__ == "__main__":
    print(f'Learning rate: {learning_rate}, Max_epoch: {Max_epoch}')
    print(f'Model: {Model_name}, Batch size: {batch_size}')
    print(f'Device: {device}')
    print(f'Training set size: {train_set}')

    Whole_Loss = []

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"📊 Training log for {Model_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Max epochs: {Max_epoch}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training started at: {datetime.datetime.now()}\n")
        f.write("="*50 + "\n\n")

        for epoch in range(1, Max_epoch + 1):
            print(f"\n=== Epoch {epoch}/{Max_epoch} ===")
            f.write(f"\n=== Epoch {epoch}/{Max_epoch} ===\n")

            # 模型训练模式
            model.train()
            train_loader = DataLoader(
                Dataset, 
                num_workers=8, 
                batch_size=batch_size, 
                shuffle=True,
                # pin_memory=True  # 加速 GPU 数据传输
            )

            # 初始化每轮损失累加器
            epoch_loss = 0.0
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0

            start_time = time.time()

            # 8. 数据迭代：关键修复！正确解包 4 个元素 (hsi, inputs, labels, mask)
            for i, (hsi, inputs, labels, mask) in enumerate(train_loader):
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

                # 9. 输入归一化：与 MST 训练和 pub_test 完全一致
                if len(inputs.shape) == 3:  # [bs, H, W]（单通道）
                    inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                else:  # [bs, C, H, W]（多通道）
                    inputs_norm = inputs.clone()
                    for c in range(inputs.size(1)):
                        inputs_norm[:, c] = (inputs[:, c] - inputs[:, c].min()) / (
                            inputs[:, c].max() - inputs[:, c].min() + 1e-8
                        )

                # 10. 修复 mask 维度：从 [84, 256, 422] 裁剪为 [84, 256, 256]
                # 关键！与 MST 训练和 pub_test 完全一致
                if mask.shape[-1] > 256:
                    mask_fixed = mask[:, :, :, :256]
                else:
                    mask_fixed = mask

                # 11. 模型前向传播：传入 (inputs_norm, mask_fixed)
                outputs = model(inputs_norm, mask_fixed)

                # 打印形状调试（每 200 步打印一次）
                if i % 200 == 0:
                    shape_log = f"Step {i}: Input shape={inputs.shape}, Mask shape={mask_fixed.shape}, Output shape={outputs.shape}, Label shape={labels.shape}"
                    print(shape_log)
                    f.write(f"{shape_log}\n")

                # 12. 标签归一化：关键！使用 XA_max_min_norm（公共数据集专用）
                # 与 pub_test 完全一致，保证训练测试归一化方法统一
                labels_norm = XA_max_min_norm(labels)

                # 13. 计算损失：与 MST 训练完全一致（混合损失）
                Charbonnier_loss = loss_CharbonnierLoss(outputs, labels_norm)
                L1_loss = loss_L1(outputs, labels_norm)
                SSIM_loss = loss_SSIM(outputs, labels_norm)
                loss = Charbonnier_loss + L1_loss + SSIM_loss

                # 累加损失
                epoch_loss += loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()

                # 14. 反向传播：与 MST 训练完全一致
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                clip_grad_norm_(model.parameters(), max_norm=0.2)

                optimizer.step()
                scheduler.step()

                # 15. 日志记录：每 50 步记录一次（与 MST 训练一致）
                if i % 50 == 0:
                    avg_total_loss = epoch_loss / (i + 1)
                    avg_charbonnier = epoch_Charbonnier_loss / (i + 1)
                    avg_l1 = epoch_L1_loss / (i + 1)
                    avg_ssim = epoch_SSIM_loss / (i + 1)

                    log_str = (
                        f"Epoch {epoch:4d} | Step {i:4d}/{train_set//batch_size:4d} | "
                        f"Whole Loss: {avg_total_loss:.6f} | "
                        f"Charbonnier: {avg_charbonnier:.6f} | "
                        f"L1: {avg_l1:.6f} | "
                        f"SSIM Loss: {avg_ssim:.6f} | "
                        f"Time: {datetime.datetime.now()}\n"
                    )
                    f.write(log_str)
                    f.flush()
                    print(log_str.strip())

            # 16. 每轮训练结束：计算平均损失与耗时
            elapsed_time = time.time() - start_time
            epoch_avg_loss = epoch_loss / (train_set // batch_size)
            epoch_avg_charbonnier = epoch_Charbonnier_loss / (train_set // batch_size)
            epoch_avg_l1 = epoch_L1_loss / (train_set // batch_size)
            epoch_avg_ssim = epoch_SSIM_loss / (train_set // batch_size)

            epoch_summary = (
                f"\n=== Epoch {epoch} Summary ===\n"
                f"Average Whole Loss: {epoch_avg_loss:.6f}\n"
                f"Average Charbonnier Loss: {epoch_avg_charbonnier:.6f}\n"
                f"Average L1 Loss: {epoch_avg_l1:.6f}\n"
                f"Average SSIM Loss: {epoch_avg_ssim:.6f}\n"
                f"Total Time: {elapsed_time:.2f} s\n"
                f"Time per Step: {elapsed_time/(train_set//batch_size):.4f} s\n"
                "="*50 + "\n"
            )
            f.write(epoch_summary)
            f.flush()
            print(epoch_summary.strip())

            # 记录 L1 损失（用于绘图）
            Whole_Loss.append(epoch_avg_l1)

            # 17. 保存模型检查点：每 50 轮或最后一轮保存（与 MST 一致）
            if epoch % 50 == 0 or epoch == Max_epoch:
                model_save_path = os.path.join(
                    result_dir, 
                    f"{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{epoch}_{train_set}_Public.pth"
                )
                torch.save(model, model_save_path)
                save_log = f"✅ Model saved at epoch {epoch}: {model_save_path}\n"
                f.write(save_log)
                f.flush()
                print(save_log.strip())

        # 18. 训练完全结束：写入最终总结
        final_summary = (
            "\n" + "="*50 + "\n"
            "📋 Training Completed!\n"
            "="*50 + "\n"
            f"Total Epochs: {Max_epoch}\n"
            f"Final L1 Loss: {Whole_Loss[-1]:.6f}\n"
            f"Minimum L1 Loss: {np.min(Whole_Loss):.6f}\n"
            f"Training Completed at: {datetime.datetime.now()}\n"
            "="*50 + "\n"
        )
        f.write(final_summary)
        print(final_summary.strip())

    # 19. 保存最终模型（与 MST 一致）
    final_model_path = os.path.join(
        result_dir, 
        f"{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{Max_epoch}_{train_set}_hybridLoss_PUBLIC.pth"
    )
    torch.save(model, final_model_path)
    print(f"💾 Final model saved: {final_model_path}")

    # 20. 保存损失数据与绘图
    # 保存 Loss.csv
    loss_csv_path = os.path.join(result_dir, f"{Model_name}_training_Loss.csv")
    np.savetxt(loss_csv_path, Whole_Loss, delimiter=",", header="L1_Loss", comments="")
    print(f"💾 Saved loss data to: {loss_csv_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, len(Whole_Loss) + 1)
    plt.plot(epoch_range, Whole_Loss, 'b-', linewidth=2, label='Training L1 Loss')
    plt.scatter(epoch_range, Whole_Loss, color='#2E86AB', s=30, alpha=0.6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.title(f'{Model_name} Training Loss Curve', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()

    # 保存损失图
    loss_plot_path = os.path.join(result_dir, f"{Model_name}_loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved loss plot to: {loss_plot_path}")

    # 21. 最终提示：告知所有结果的保存位置
    print("\n" + "="*60)
    print("🎉 Training Completed Successfully!")
    print(f"📁 All Results Saved to: {os.path.abspath(result_dir)}")
    print(f"📄 Log File: {os.path.abspath(log_file)}")
    print(f"📊 Loss Plot: {os.path.abspath(loss_plot_path)}")
    print(f"💾 Final Model: {os.path.abspath(final_model_path)}")
    print("="*60)