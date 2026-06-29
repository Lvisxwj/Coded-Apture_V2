# CASSI-VegIndex

本项目研究从 CASSI 二维压缩测量直接预测 32 个植被指数，避免先完整重构
84 波段高光谱立方体再逐指数计算。当前可复用的训练产物只有 Restormer
占位基线；MST-Mamba、WPO3D、DHM 和 IFGNet 均处于代码验证阶段。

## 活动入口

- `train.py`：统一训练入口，模型与设备在 `cfg.py` 选择。
- `test.py`：确定性整叶测试，保存 patch、指标和 `test_meta.json`。
- `viz.py`：按元数据拼接 patch，裁掉 padding 后生成完整叶片图。
- `config.yaml`：服务器路径、数据划分和模型超参数。

服务器项目位于 `/data5/SCI/vegindex`。同机的
`/data5/SCI/xieweijie/CASSI` 是另一个项目，不属于本仓库。

## 当前状态

- 数据：252 组连续编号 HSI/标签和 1 个 mask。
- Restormer：完成 300 epoch；epoch 300 是已保存节点中训练损失最低者。
- 旧测试结果：因测试脚本重复归一化而失真，已经归档，禁止用于论文结论。
- 新测试脚本：指标统一在固定的 32 通道归一化空间计算。

具体命令见 `docs/command.md`，目录迁移见 `docs/directory_map.md`，评估结论见
`docs/evaluation_report.md`。
