"""
cfg.py — 项目总开关。修改这里来切换模型和关键配置。

train.py / test.py 都从这里读取模型选择、GPU、checkpoint 路径。
"""

# ── 模型选择 ──
MODELS = ["mst_mamba", "wpo3d", "restormer", "dhm", "ifgnet"]
MODEL_IDX = 2                          # ← 改这个数字切模型（0-4）
MODEL_NAME = MODELS[MODEL_IDX]

# ── GPU ──
GPU_ID = "cuda:3"

# ── Checkpoint（test.py 使用）──
BEST_CKPT = (                          # 按训练 loss 选出的占位权重，非验证集 best
    "/data5/SCI/vegindex/code/result/model/restormer/"
    "restormer_ep300_lr1e-04_bs1.pth"
)

# ── 32 个植被指数名称（与 Norm.py 通道顺序一致）──
INDEX_NAMES = [
    "DD", "TVI", "LCI", "mND680", "mND705", "PSSR", "CRI550", "CRI700",
    "MCARI", "SAVI", "CI_green", "CI_red_edge", "NDVI", "DVI", "ATSAVI",
    "EVI", "GI", "MSAVI", "MSR", "MVTI1", "MVTI2", "OSAVI", "PSND",
    "RDVI", "SPVI", "TCARI", "SR", "VARI_green", "WDRVI", "ARI", "BGI", "BRI",
]
