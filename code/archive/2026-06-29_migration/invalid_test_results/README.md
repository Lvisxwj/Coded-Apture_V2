# 旧 Restormer 测试结果（不可用于论文）

旧 `test_unified.py` 在模型已经输出归一化指数后，又调用一次固定通道
`max_min_norm(outputs)`，导致 CRI、ARI 等通道出现数百量级 L1 和负 PSNR。
本目录仅保留原始证据。待 GPU 空闲后必须使用活动 `test.py` 重新评估。
