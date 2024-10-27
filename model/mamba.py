from common import *
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

class SingleMambaBlock(nn.Module):
    """Single input Mamba"""
    def __init__(self, dim, H, W):
        super().__init__()
        self.H=H
        self.W=W
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)


    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm(input)
        output = self.block(input)
        output = rearrange(output, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        # output = self.norm1(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    """Two input Mamba"""
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        return output + skip


class FusionMamba(nn.Module):
    """Two Mamba fusion"""
    def __init__(self, dim, H, W, depth=1):
        super().__init__()
        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])
        #_ 是一个常见的惯用语法，用作临时变量的名称，虽然变量 _ 赋值后可能不会在循环体内使用，但在 Python 中，使用 _ 作为变量名是一种告诉其他读者这个变量实际上不会被用到的方式。
        for _ in range(depth):
            self.spa_mamba_layers.append(SingleMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(SingleMambaBlock(dim, H, W))
        self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
        self.spe_cross_mamba = CrossMambaBlock(dim, H, W)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        pan = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
        ms = rearrange(ms, 'b c h w -> b (h w) c', h=h, w=w)
        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            pan = spa_layer(pan)
            ms = spe_layer(ms)
        spa_fusion = self.spa_cross_mamba(pan, ms)
        spe_fusion = self.spe_cross_mamba(ms, pan)
        fusion = self.out_proj((spa_fusion + spe_fusion) / 2)
        pan = rearrange(pan, 'b (h w) c -> b c h w', h=h, w=w)
        ms = rearrange(ms, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        return pan, ms + output