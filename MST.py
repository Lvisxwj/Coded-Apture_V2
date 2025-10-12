import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

# Tensor = 1.1

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)
        return mask_emb

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mm = MaskGuidedMechanism(dim)
        self.dim = dim

    def forward(self, x_in, mask=None):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        mask_attn = self.mm(mask.permute(0,3,1,2)).permute(0,2,3,1)
        if b != 0:
            mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])
        q, k, v, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))
        v = v * mask_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, mask=mask.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(self, dim=84, stage=3, num_blocks=[2,2,2]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        # self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)
        # HSI dataset
        self.embedding = nn.Conv2d(84, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),   #????????
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 84, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if mask == None:
            mask = torch.zeros((1,84,256,310)).cuda()

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks = []
        for (MSAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = MSAB(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = LeWinBlcok(fea, mask)

        # Mapping
        out = self.mapping(fea) + x

        return out




# # coded-apture/models_mst/architecture/MST.py
# import torch
# import torch.nn as nn
# from .MSAB import MSAB  # 后续会迁移MSAB模块

# class MST(nn.Module):
#     def __init__(self, dim=64, stage=3, num_blocks=[2,2,2], in_channels=1, out_channels=3):
#         super(MST, self).__init__()
#         self.dim = dim
#         self.stage = stage

#         ###############通道#####################
#         # 输入投影层：将1通道测量值映射到dim维度（原MST是28通道，这里修改为1通道输入）
#         self.embedding = nn.Conv2d(in_channels, self.dim, 3, 1, 1, bias=False)

#         # 编码器：保留多尺度结构，调整通道数变化逻辑
#         self.encoder_layers = nn.ModuleList([])
#         dim_stage = dim
#         for i in range(stage):
#             self.encoder_layers.append(nn.ModuleList([
#                 MSAB(
#                     dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
#                 # 特征下采样：通道数翻倍
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
#                 # 掩码下采样（如果coded-apture有掩码，保持与特征相同的下采样方式）
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
#             ]))
#             dim_stage *= 2

#         # 瓶颈层：多尺度注意力模块
#         self.bottleneck = MSAB(
#             dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

#         # 解码器：与编码器对称，通道数减半
#         self.decoder_layers = nn.ModuleList([])
#         for i in range(stage):
#             self.decoder_layers.append(nn.ModuleList([
#                 # 特征上采样：通道数减半
#                 nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
#                 # 特征融合：拼接后降维
#                 nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
#                 # 解码器注意力模块
#                 MSAB(
#                     dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
#                     heads=(dim_stage // 2) // dim),
#             ]))
#             dim_stage //= 2

#         # 输出投影层：将dim维度映射到3通道RGB（原MST是28通道，这里修改为3通道输出）
#         self.mapping = nn.Conv2d(self.dim, out_channels, 3, 1, 1, bias=False)

#         # 激活函数（沿用MST的LeakyReLU）
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def forward(self, x, mask=None):
#         """
#         x: [b, 1, h, w]  # coded-apture的输入：1通道编码测量值
#         mask: [b, 1, h, w]  # 可选：coded-apture的编码模板（如果需要）
#         return out: [b, 3, h, w]  # 输出3通道RGB图像
#         """
#         # 处理掩码（如果未传入，初始化与输入同形状的零掩码）
#         if mask is None:
#             mask = torch.zeros_like(x).cuda()  # 形状与x一致：[b,1,h,w]

#         # 输入嵌入：将1通道映射到dim维度
#         fea = self.lrelu(self.embedding(x))

#         # 编码器流程：多尺度特征提取+掩码下采样
#         fea_encoder = []  # 保存编码器各阶段特征
#         masks = []  # 保存各阶段掩码
#         for (msab, fea_down, mask_down) in self.encoder_layers:
#             fea = msab(fea, mask)  # 注意力模块（结合特征和掩码）
#             masks.append(mask)
#             fea_encoder.append(fea)
#             fea = fea_down(fea)  # 特征下采样
#             mask = mask_down(mask)  # 掩码下采样

#         # 瓶颈层处理
#         fea = self.bottleneck(fea, mask)

#         # 解码器流程：上采样+特征融合
#         for i, (fea_up, fusion, msab) in enumerate(self.decoder_layers):
#             fea = fea_up(fea)  # 特征上采样
#             # 与编码器对应阶段的特征拼接后融合
#             fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
#             mask = masks[self.stage - 1 - i]  # 恢复对应阶段的掩码
#             fea = msab(fea, mask)  # 解码器注意力模块

#         # 输出层：残差连接（原输入+重建特征）
#         out = self.mapping(fea) + x

#         return out