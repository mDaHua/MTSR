from numpy.core.shape_base import block
from twisted.web.html import output

from common import *
from einops import rearrange
from mamba_ssm.modules.mamba_simple1 import Mamba
from pytorch_wavelets import DWTForward, DWTInverse
from model_4.transformer import SA

device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)


class SingleMambaBlock(nn.Module):
    """Single input Mamba"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
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


class EncoderMambaBlock(nn.Module):
    """mamba encoder block"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.norm1 = nn.LayerNorm(dim)
        self.sdmamba = SDMambaBlock(dim, H, W)
        self.norm2 = nn.LayerNorm(dim)
        self.pffn = PoswiseFeedForwardNet(dim, H, W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input

        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)

        output = self.sdmamba(input)
        input = output + skip

        skip = input

        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm2(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)

        output = self.pffn(input)
        output = output + skip
        return output


class ShiftMambaBlock(nn.Module):
    """mamba's input changes to H*C*W and W*C*H"""

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.C = dim
        self.H = H
        self.W = W

        self.block_1 = Mamba(H, expand=1, d_state=8, bimamba_type='v6',
                             if_devide_out=True, use_norm=True, input_h=dim, input_w=W)
        self.block_2 = Mamba(W, expand=1, d_state=8, bimamba_type='v6',
                             if_devide_out=True, use_norm=True, input_h=dim, input_w=H)
        self.conv = nn.Sequential(
            conv(dim, dim, 3),
            nn.ReLU()
        )

    def forward(self, input):
        # input: (B, N, C)
        skip = input

        input_1 = rearrange(input, 'b c h w -> b h c w')
        input_2 = rearrange(input, 'b c h w -> b w c h')

        input_1 = rearrange(input_1, 'b h c w -> b (c w) h', c=self.C, w=self.W)
        input_2 = rearrange(input_2, 'b w c h -> b (c h) w', c=self.C, h=self.H)

        output_1 = self.block_1(input_1)
        output_2 = self.block_2(input_2)

        output_1 = rearrange(output_1, 'b (c w) h -> b h c w', c=self.C, w=self.W)
        output_2 = rearrange(output_2, 'b (c h) w -> b w c h', c=self.C, h=self.H)

        output_1 = rearrange(output_1, 'b h c w -> b c h w')
        output_2 = rearrange(output_2, 'b w c h -> b c h w')

        output = self.conv(output_1 + output_2)

        return output + skip


class DWTMambaBlock(nn.Module):
    """Discrete wavelet transform mamba"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.C = dim
        self.H = H
        self.W = W

        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
                              if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        b, c, h, w = input.shape
        skip = input

        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)

        output = self.block(input)

        output = rearrange(output, 'b (h w) c -> b c h w', h=self.H, w=self.W)

        return output + skip


class PoswiseFeedForwardNet(nn.Module):
    """Feedforward network"""

    def __init__(self, dim, H, W):
        super(PoswiseFeedForwardNet, self).__init__()
        self.H = H
        self.W = W
        self.fc = nn.Sequential(
            nn.Linear(dim, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, dim, bias=False)
        )

    def forward(self, input):
        # inputs: [batch_size, seq_len, d_model]
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        output = self.fc(input)
        output = rearrange(output, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        return output


class SDMambaBlock(nn.Module):
    """ShiftMambaBlock and DWTMambaBlock"""

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.H = H
        self.W = W
        self.ca = CALayer(dim, 16)
        self.ca1 = CALayer1(dim, 16)
        self.sa = SA(dim, 8, False)
        self.DWTMambaBlock = DWTMambaBlock(dim, H, W)
        self.conv = nn.Sequential(
            conv(dim, dim, 3),
        )

    def forward(self, input):
        # output1 = self.ShiftMambaBlock(input)
        output1 = self.ca(input)
        # output1 = self.ca1(input)
        # output1 = self.sa(input)
        output2 = self.DWTMambaBlock(input)
        output = self.conv(output1 + output2)
        # output = output1 * output2
        return output


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
        # _ 是一个常见的惯用语法，用作临时变量的名称，虽然变量 _ 赋值后可能不会在循环体内使用，但在 Python 中，使用 _ 作为变量名是一种告诉其他读者这个变量实际上不会被用到的方式。
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
