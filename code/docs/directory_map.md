# 本地—服务器目录映射

| 内容 | 本地 | 服务器 | 同步策略 |
|---|---|---|---|
| 活动代码 | `code/` | `/data5/SCI/vegindex/code/` | 修改后 SHA-256 对齐 |
| 项目数据 | 不下载 | `/data5/SCI/vegindex/data/` | 仅服务器，约 110 GB |
| 周期权重 | 不全量下载 | `code/result/model/restormer/` | 6 个权重保留服务器 |
| 占位权重 | `code/artifacts/checkpoints/restormer/best_train.pth` | 原始 epoch 300 权重 | 本地保存一份并校验哈希 |
| 论文 | `latex/` | 不上传 | 仅本地维护 |
| 历史代码 | `code/archive/2026-06-29_migration/` | 同相对目录 | 只归档、不删除 |
| 修改前备份 | `backups/2026-06-29_pre_cleanup/` | 同相对目录 | 两端独立保留 |

同一服务器上的 `/data5/SCI/xieweijie/CASSI` 是 AAAI/CASSI 项目，明确排除在
vegindex 的整理、同步和论文任务之外。

## 活动与归档边界

活动代码只保留 `train.py`、`test.py`、`viz.py`、数据/损失/归一化模块、五个候选
模型、assets、tests 和当前文档。重复入口、旧 MST baseline、历史生成文档、
溢出日志及旧测试结果全部进入日期归档目录。
