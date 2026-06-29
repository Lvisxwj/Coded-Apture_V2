# 项目架构

## 任务与数据流

输入为 CASSI 测量 `Y in R^(H x (W+166))`，对应 84 个波段、色散步长 2；
输出为 `I_hat in R^(32 x H x W)`。活动数据流为：

1. `dataset_npy.py` 从连续编号的 HSI/label 中裁剪 patch，并模拟 CASSI 测量。
2. `train.py::normalize_input` 对每个样本独立归一化测量值。
3. 模型把测量值展开为 84 通道初始特征，并预测 32 通道归一化指数。
4. `Norm.py` 使用唯一一组固定通道上下界归一化标签或恢复物理量纲。
5. `test.py` 在归一化空间计算 L1/PSNR，保存原量纲 patch 和 JSON 元数据。
6. `viz.py` 拼接 patch，并按原图尺寸裁去底部和右侧 padding。

## 模型状态

| 模型 | 入口 | 状态 |
|---|---|---|
| Restormer | `models/restormer` | 已训练，占位基线 |
| MST-Mamba | `models/mst_mamba` | 待正式训练 |
| WPO3D | `models/wpo3d` | 待正式训练 |
| DHM | `models/dhm` | 待正式训练 |
| IFGNet | `models/ifgnet` | 待正式训练 |

五个模型均遵循 `forward(measurement, mask)` 接口；IFGNet 额外返回公式分支和
光谱重建分支，训练/测试入口只使用融合输出作为主预测。

## 关键约束

- 活动 NPY 编号为 1--252，不再跳过原始 MAT 的 15/43/109。
- 当前 Restormer checkpoint 来自旧划分，实际训练源场景为 197 个；其配置快照
  保存在 `artifacts/checkpoints/restormer/legacy_training_config.yaml`。
- `best_train.pth` 只表示已保存节点中的最低训练损失，不等价于验证集最优。
- 新的正式数值必须由修订后的 `test.py` 重新产生。
