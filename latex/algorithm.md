# 算法与实验定义

## CASSI 前向模型

- HSI：`[H, W, 84]`，445--860 nm，间隔 5 nm。
- Mask：`[H, W]`，扩展到 84 通道。
- 第 `l` 个波段沿宽度方向移动 `2*l` 个像素后求和。
- patch 为 256 时测量尺寸为 `[256, 422]`。
- 当前实现还执行 `/84 * 0.9` 的尺度变换。

## 直接任务

`prediction = model(normalize(measurement), shifted_mask)`，输出
`[batch, 32, H, W]`。标签和指标使用 `Norm.py` 中唯一一组固定通道上下界。

## 32 指数

活动通道顺序为 DD、TVI、LCI、mND680、mND705、PSSR、CRI550、CRI700、
MCARI、SAVI、CI-green、CI-red-edge、NDVI、DVI、ATSAVI、EVI、GI、MSAVI、
MSR、MVTI1、MVTI2、OSAVI、PSND、RDVI、SPVI、TCARI、SR、VARI-green、
WDRVI、ARI、BGI、BRI。精确公式以 `code/assets/index.csv` 和 `label.m` 为核验源。

## 统一评估

- 主指标：归一化空间逐通道 L1、PSNR，以及跨通道宏平均。
- 可选指标：SSIM、原量纲逐通道误差、参数量、推理时间、峰值显存。
- 禁止把不同物理量纲的原始误差直接平均后作为唯一结论。
- 整叶结果必须记录原始尺寸并裁去 padding。
