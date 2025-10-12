# coded-apture/models_mst/architecture/MSAB.py
import torch
import torch.nn as nn
from einops import rearrange  # 需要安装einops：pip install einops

class MSAB_fuck(nn.Module):
    def __init__(self, dim, num_blocks, dim_head, heads):
        super(MSAB_fuck, self).__init__()
        self.blocks = nn.ModuleList([
            # 每个block包含注意力和前馈网络（根据MST原代码实现）
            nn.Sequential(
                nn.LayerNorm(dim),
                Attention(dim, dim_head, heads),  # 假设已有Attention类
                nn.LayerNorm(dim),
                nn.Conv2d(dim, dim * 4, 1),
                nn.GELU(),
                nn.Conv2d(dim * 4, dim, 1)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, mask):
        """
        x: [b, dim, h, w]  # 特征
        mask: [b, 1, h, w]  # 掩码（用于引导注意力）
        """
        for block in self.blocks:
            # 注意力模块结合掩码（具体逻辑根据MST原实现调整）
            x = x + block[0](x)  # 注意力残差
            x = x + block[3:](block[2](x))  # 前馈网络残差
        return x

# 补充Attention类（根据MST中注意力实现迁移）
class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        self.to_qkv = nn.Conv2d(dim, dim_head * heads * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim_head * heads, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) h w -> b h (h w) d', h=self.heads), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h (h w) d -> b (h d) h w', h=h, w=w)
        return self.to_out(out)