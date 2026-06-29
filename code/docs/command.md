# 运行命令

以下命令均在服务器执行。当前 GPU 被占用时只执行 CPU 冒烟检查，不运行训练或
真实数据测试。

```bash
ssh cassi-server
cd /data5/SCI/vegindex/code
conda activate zwj
```

## CPU 冒烟检查

```bash
CUDA_VISIBLE_DEVICES='' python tests/smoke_cpu.py --utilities-only
for model in mst_mamba wpo3d restormer dhm ifgnet; do
  CUDA_VISIBLE_DEVICES='' python tests/smoke_cpu.py --model "$model"
done
```

## 训练与测试（仅在 GPU 获准且空闲时）

在 `cfg.py` 选择模型、GPU 和 checkpoint，在 `config.yaml` 设置超参数后：

```bash
python train.py
python test.py
```

`test.py` 输出目录包含 `preds/`、`labels/`、`metrics_summary.csv`、
`test_meta.json` 和逐通道指标图。把该目录填入 `viz.py::TEST_RESULT_DIR` 后运行：

```bash
python viz.py
```

旧的 `train_unified.py`、`test_unified.py` 和 `visualization.py` 已归档，不再使用。
