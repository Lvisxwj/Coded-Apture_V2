from torch.autograd import Variable
import sys
import os
# 1. 统一结果目录：创建 Pub_test_result，所有输出放这里
result_dir = "./Pub_test_result"
os.makedirs(result_dir, exist_ok=True)
sys.path.append("/data4/zhuo-file/fyc_file/ATTU_Test/")

import torch
from torch.utils.data import DataLoader
from Pub_dataset import dataset  # 确保 Pub_dataset 与 MST 数据集逻辑一致
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  # 用于保存测试样本的 .mat 文件
from loss import CharbonnierLoss, SSIM  # 明确导入损失函数（避免依赖 * 导入）
from torch.nn.utils import clip_grad_norm_
from Norm import XA_max_min_norm  # 公共数据集专用归一化函数


# 2. 数据路径与加载：确保与 MST 测试逻辑一致（按模式隔离数据）
# 注意：需确保 Pub_dataset 的 dataset 类支持 is_train 参数（参考 MST_dataset_corrected 的修改）
hsi_filename = r"/data4/zhuo-file/extracted_data/Public_Data/XiongAn_jg5.mat"
label_filename = r"/data4/zhuo-file/extracted_data/Public_Data/new_XiongAn_indices.mat"
# 加载测试集：明确传入 is_train=False，避免加载训练数据（关键！防止数据泄露）
Dataset = dataset(hsi_filename, label_filename, is_train=False)


# 3. GPU 与模型加载：适配 MST 模型的输入要求（若模型需要 mask，需补充）
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_path = r"/home/graduate/lizhuo/xwj/parameter/ours/MST_Corrected_MST_Corrected_lr1e-04_bs4_Epoch500_2560_hybridLoss.pth"

# 加载模型：增加异常处理（避免模型路径错误导致代码崩溃）
try:
    model = torch.load(model_path, map_location=device)
    print(f"✅ Loaded MST model from: {model_path}")
except Exception as e:
    print(f"❌ Could not load model from {model_path}, error: {str(e)}")
    print("⚠️ Initializing new MST_Corrected model (if MST_corrected is in path)")
    # 若模型加载失败，初始化新模型（需确保 MST_corrected 能导入）
    from MST_corrected import MST_Corrected
    model = MST_Corrected(dim=64, stage=3, num_blocks=[2,2,2])
model.to(device)
model.eval()  # 测试前先设为评估模式（禁用 Dropout 等）


# 4. 测试超参数：与 MST 测试保持一致（避免批次大小、迭代次数不匹配）
batch_size = 1
Max_epoch = 1  # 测试集通常只跑 1 轮（无需多轮迭代，避免重复计算）
test_set = 2560  # 需与 Pub_dataset 的 __len__ 输出一致（或用 len(Dataset) 自动获取）
save_sample_num = 5  # 保存前 5 个测试样本的结果（用于可视化分析）


# 5. 损失函数定义：与 MST 测试完全一致（保证评估标准统一）
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()


if __name__ == "__main__":
    Whole_Loss = []  # 记录每轮的平均损失
    # 统一日志路径：放到结果目录下
    log_file = os.path.join(result_dir, "Pub_testing_log.txt")
    # 样本保存子目录：单独存放可视化结果和 .mat 文件
    sample_dir = os.path.join(result_dir, "test_samples")
    os.makedirs(sample_dir, exist_ok=True)

    # 打开日志文件：写入测试基本信息（方便后续追溯）
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("📊 Public Dataset (XiongAn) Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Device: {device}\n")
        f.write(f"Test Set Size: {test_set}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Start Time: {datetime.datetime.now()}\n")
        f.write("="*50 + "\n\n")

        # 测试迭代（Max_epoch=1，仅跑 1 轮）
        for epoch in range(1, Max_epoch + 1):
            print(f"\n=== Test Epoch {epoch}/{Max_epoch} (Device: {device}) ===")
            f.write(f"\n=== Test Epoch {epoch}/{Max_epoch} ===\n")

            # 初始化损失累加器（每轮重新清零）
            epoch_loss = 0.0
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0
            start_time = time.time()

            # 加载测试数据：num_workers=8（与 MST 一致），shuffle=False（测试不打乱）
            test_loader = DataLoader(
                Dataset, 
                num_workers=8, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True  # 加速 GPU 数据传输（可选，大样本时有效）
            )

            # 存储测试样本（用于后续保存）
            all_outputs = []
            all_labels = []
            all_inputs = []  # 存储输入用于对比


            # 6. 数据迭代：适配 MST 模型的输入（关键！若模型需要 mask，需补充 unpack）
            # 注意：若 MST 模型需要 (inputs_norm, mask) 两个输入，需修改此处的 unpack 逻辑
            # 假设 Pub_dataset 返回 (hsi, inputs, labels, mask)（与 MST 一致），若仅返回 (inputs, labels)，需调整
            for i, (hsi, inputs, labels, mask) in enumerate(test_loader):
                # 数据移到 GPU（跳过未使用的 hsi，不影响运行）
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                all_inputs.append(inputs.cpu().numpy())  # 保存输入用于后续分析

                # 7. 输入归一化：与 MST 训练时完全一致（关键！保证输入分布匹配）
                # 处理 inputs 形状：若为 [bs, H, W]（2D），按通道归一化；若为 [bs, C, H, W]（3D），按通道归一化
                if len(inputs.shape) == 3:  # [bs, H, W]（单通道）
                    inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                else:  # [bs, C, H, W]（多通道）
                    inputs_norm = inputs.clone()
                    for c in range(inputs.size(1)):
                        inputs_norm[:, c] = (inputs[:, c] - inputs[:, c].min()) / (
                            inputs[:, c].max() - inputs[:, c].min() + 1e-8
                        )

                # 8. 适配 MST 模型的 mask 输入（关键！若模型需要 mask，需裁剪形状）
                # 参考 MST 测试逻辑：mask 从 [84, 256, 422] 裁剪为 [84, 256, 256]
                if mask.shape[-1] > 256:  # 若 mask 宽度超过 256，裁剪
                    mask_fixed = mask[:, :, :, :256]
                else:
                    mask_fixed = mask


                # 9. 模型推理：禁用梯度计算（测试阶段必须！否则内存溢出）
                with torch.no_grad():
                    # 注意：MST 模型需要两个输入（inputs_norm, mask_fixed），若只传 inputs 会报错
                    outputs = model(inputs_norm, mask_fixed)  # 关键！与 MST 模型输入匹配

                # 打印形状调试（方便确认输入输出是否匹配）
                if i % 200 == 0:  # 每 200 步打印一次（避免日志过多）
                    shape_log = f"Step {i}: Input shape={inputs.shape}, Mask shape={mask_fixed.shape}, Output shape={outputs.shape}, Label shape={labels.shape}"
                    print(shape_log)
                    f.write(f"{shape_log}\n")


                # 10. 结果归一化与损失计算：与 MST 测试完全一致
                # 公共数据集用 XA_max_min_norm，确保与训练时的归一化函数匹配
                outputs_norm = XA_max_min_norm(outputs)
                labels_norm = XA_max_min_norm(labels)

                # 计算损失（与 MST 一致：混合损失 = Charbonnier + L1 + SSIM）
                Charbonnier_loss = loss_CharbonnierLoss(outputs_norm, labels_norm)
                L1_loss = loss_L1(outputs_norm, labels_norm)
                SSIM_loss = loss_SSIM(outputs_norm, labels_norm)
                total_loss = Charbonnier_loss + L1_loss + SSIM_loss

                # 累加损失（用于计算平均损失）
                epoch_loss += total_loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()

                # 保存输出和标签（用于后续保存样本）
                all_outputs.append(outputs_norm.cpu().numpy())
                all_labels.append(labels_norm.cpu().numpy())


                # 11. 日志记录：每 100 步记录一次（与 MST 一致）
                if i % 100 == 0:
                    # 计算当前平均损失
                    avg_total_loss = epoch_loss / (i + 1)
                    avg_charbonnier = epoch_Charbonnier_loss / (i + 1)
                    avg_l1 = epoch_L1_loss / (i + 1)
                    avg_ssim = epoch_SSIM_loss / (i + 1)

                    # 日志字符串（格式与 MST 一致，方便对比）
                    log_str = (
                        f"Epoch {epoch:4d} | Step {i:4d}/{test_set//batch_size:4d} | "
                        f"Whole Loss: {avg_total_loss:.6f} | "
                        f"Charbonnier: {avg_charbonnier:.6f} | "
                        f"L1: {avg_l1:.6f} | "
                        f"SSIM Loss: {avg_ssim:.6f} | "
                        f"Time: {datetime.datetime.now()}\n"
                    )
                    # 写入日志并打印
                    f.write(log_str)
                    f.flush()  # 实时写入（避免日志缓存）
                    print(log_str.strip())


                # 12. 终止条件：若测试样本数达到 test_set，提前退出（避免多跑）
                if i + 1 >= test_set // batch_size:
                    print(f"✅ Reached test set size ({test_set}), stop early")
                    f.write(f"Reached test set size ({test_set}), stop early\n")
                    break


            # 13. 每轮测试结束：计算平均损失与耗时
            epoch_avg_loss = epoch_loss / (i + 1)
            epoch_avg_charbonnier = epoch_Charbonnier_loss / (i + 1)
            epoch_avg_l1 = epoch_L1_loss / (i + 1)
            epoch_avg_ssim = epoch_SSIM_loss / (i + 1)
            elapsed_time = time.time() - start_time

            # 写入轮次总结日志
            epoch_summary = (
                f"\n=== Epoch {epoch} Summary ===\n"
                f"Average Whole Loss: {epoch_avg_loss:.6f}\n"
                f"Average Charbonnier Loss: {epoch_avg_charbonnier:.6f}\n"
                f"Average L1 Loss: {epoch_avg_l1:.6f}\n"
                f"Average SSIM Loss: {epoch_avg_ssim:.6f}\n"
                f"Total Time: {elapsed_time:.2f} s\n"
                f"Time per Step: {elapsed_time/(i+1):.4f} s\n"
                "="*50 + "\n"
            )
            f.write(epoch_summary)
            f.flush()
            print(epoch_summary.strip())

            # 记录全局损失（用于后续绘图）
            Whole_Loss.append(epoch_avg_l1)  # 记录 L1 损失（常用作主要指标）


            # 14. 保存测试样本：与 MST 一致，保存前 N 个样本的 .mat 和可视化图
            print(f"💾 Saving first {save_sample_num} test samples to {sample_dir}")
            f.write(f"Saving first {save_sample_num} test samples to {sample_dir}\n")

            for idx in range(min(save_sample_num, len(all_outputs))):
                # 提取单个样本（batch_size=1，取第 0 个）
                input_sample = all_inputs[idx][0]  # [C, H, W] 或 [H, W]
                output_sample = all_outputs[idx][0]  # [4, 64, 64]（公共数据集输出）
                label_sample = all_labels[idx][0]    # [4, 64, 64]（公共数据集标签）

                # 保存 .mat 文件（方便后续用 MATLAB 分析）
                mat_path = os.path.join(sample_dir, f"Pub_sample_{idx}_results.mat")
                sio.savemat(
                    mat_path,
                    {
                        "input": input_sample,
                        "output": output_sample,
                        "label": label_sample,
                        "method": "MST_Corrected",
                        "epoch": epoch,
                        "l1_loss": loss_L1(torch.tensor(output_sample), torch.tensor(label_sample)).item()
                    }
                )

                # 可视化：与 MST 一致，用前 3 个通道做 RGB 对比图
                if output_sample.shape[0] >= 3:  # 确保有至少 3 个通道用于可视化
                    # 通道维度调整：[C, H, W] → [H, W, C]（适合 matplotlib）
                    rgb_output = np.transpose(output_sample[:3, :, :], (1, 2, 0))
                    rgb_label = np.transpose(label_sample[:3, :, :], (1, 2, 0))
                    # 输入可视化（若输入是单通道，转成 3 通道灰度图）
                    if len(input_sample.shape) == 2:  # 单通道输入 [H, W]
                        rgb_input = np.stack([input_sample]*3, axis=2)
                    else:  # 多通道输入 [C, H, W]，取第 1 通道
                        rgb_input = np.transpose(input_sample[:3, :, :], (1, 2, 0))

                    # 可视化归一化（确保像素值在 0~1 之间，避免过亮/过暗）
                    def vis_norm(x):
                        return (x - x.min()) / (x.max() - x.min() + 1e-8)
                    rgb_input = vis_norm(rgb_input)
                    rgb_output = vis_norm(rgb_output)
                    rgb_label = vis_norm(rgb_label)

                    # 绘制对比图（输入 + 输出 + 标签，三图对比）
                    plt.figure(figsize=(15, 5))
                    # 输入图
                    plt.subplot(1, 3, 1)
                    plt.imshow(rgb_input)
                    plt.title(f"Input (Sample {idx})")
                    plt.axis("off")
                    # 输出图
                    plt.subplot(1, 3, 2)
                    plt.imshow(rgb_output)
                    plt.title(f"MST Output (Sample {idx})")
                    plt.axis("off")
                    # 标签图（Ground Truth）
                    plt.subplot(1, 3, 3)
                    plt.imshow(rgb_label)
                    plt.title(f"Ground Truth (Sample {idx})")
                    plt.axis("off")

                    # 保存图片（高分辨率，方便论文使用）
                    img_path = os.path.join(sample_dir, f"Pub_sample_{idx}_comparison.png")
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=300, bbox_inches="tight")
                    plt.close()  # 关闭图片，释放内存
                    print(f"✅ Saved sample {idx}: {mat_path} | {img_path}")


        # 15. 测试完全结束：写入最终总结
        final_summary = (
            "\n" + "="*50 + "\n"
            "📋 Final Test Summary\n"
            "="*50 + "\n"
            f"Total Test Epochs: {Max_epoch}\n"
            f"Average L1 Loss Across Epochs: {np.mean(Whole_Loss):.6f}\n"
            f"Minimum L1 Loss: {np.min(Whole_Loss):.6f}\n"
            f"Test Completed Time: {datetime.datetime.now()}\n"
            "="*50 + "\n"
        )
        f.write(final_summary)
        print(final_summary.strip())


    # 16. 保存损失数据与绘图：与 MST 一致，结果放统一目录
    # 保存 Loss.csv（方便后续用 Excel 分析）
    loss_csv_path = os.path.join(result_dir, "Pub_test_Loss.csv")
    np.savetxt(loss_csv_path, Whole_Loss, delimiter=",", header="L1_Loss", comments="")
    print(f"💾 Saved loss data to: {loss_csv_path}")

    # 绘制损失图（散点 + 折线，清晰展示趋势）
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, len(Whole_Loss) + 1)
    plt.scatter(epoch_range, Whole_Loss, color="#2E86AB", s=60, label="Test L1 Loss")
    plt.plot(epoch_range, Whole_Loss, color="#A23B72", linestyle="--", linewidth=2, alpha=0.8)
    plt.xlabel("Test Epoch", fontsize=12)
    plt.ylabel("L1 Loss", fontsize=12)
    plt.title("MST Model Test Loss on Public Dataset (XiongAn)", fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle=":")
    # 保存损失图
    loss_img_path = os.path.join(result_dir, "Pub_test_Loss.png")
    plt.tight_layout()
    plt.savefig(loss_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved loss plot to: {loss_img_path}")


    # 17. 最终提示：告知所有结果的保存位置
    print("\n" + "="*60)
    print("🎉 All Public Dataset Test Completed!")
    print(f"📁 All Results Saved to: {os.path.abspath(result_dir)}")
    print(f"📄 Log File: {os.path.abspath(log_file)}")
    print(f"📊 Loss Plot: {os.path.abspath(loss_img_path)}")
    print(f"📦 Test Samples: {os.path.abspath(sample_dir)}")
    print("="*60)