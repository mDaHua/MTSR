import torch
import math
import torch.nn as nn
from common import *
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SA(nn.Module):
    """spectral attention (SA)
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, bias):
        super(SA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.deformconv(x)
        _, k, v = self.qkv(x).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class SSA(nn.Module):
    """spectral-spatial attention (SSA)"""

    def __init__(self, dim, num_heads, bias , conv=default_conv):
        super(SSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.tail = conv(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # q使用可变性卷积
        q = self.deformconv(x)
        _, k_spe, v_spe = self.qkv(x).chunk(3, dim=1)

        q_spe = rearrange(q, 'b (head c) h w -> b head c (h w)',  head=self.num_heads)
        k_spe = rearrange(k_spe, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_spe = rearrange(v_spe, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k_spe = torch.nn.functional.normalize(k_spe, dim=-1)
        v_spe = torch.nn.functional.normalize(v_spe, dim=-1)

        attn_spe = (q_spe @ k_spe.transpose(-2, -1)) * self.temperature
        attn_spe = attn_spe.softmax(dim=-1)

        out_spe = (attn_spe @ v_spe)

        out_spe = rearrange(out_spe, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out_spe

        out = self.project_out(out)
        return out


class DCTM(nn.Module):
    """  Transformer Block:deformable convolution-based transformer module (DCTM)
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU, bias=False):
        super(DCTM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.num_heads = num_heads

        self.global_attn = SSA(dim, num_heads, bias)

    def forward(self, x):

        B, C, H, W = x.shape   # B, C, H*W
        #从第3个维度开始展平，并交换2和3维度
        x = x.flatten(2).transpose(1, 2)   # B, H*W, C
        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C) # view用于改变张量形状
        x = x.permute(0, 3, 1, 2).contiguous()  # B C HW
        x = self.global_attn(x)  # global spectral self-attention

        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
