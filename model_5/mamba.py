from triton.interpreter.memory_map import torch
from twisted.web.html import output

from common import *
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
#from mamba_ssm.modules.mamba_simple1 import Mamba
from pytorch_wavelets import DWTForward, DWTInverse

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
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
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
        self.H = H
        self.W = W
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v3',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = rearrange(input0, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input1 = rearrange(input1, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        output = rearrange(output, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        # output = self.norm2(output)
        return output + skip

class CAMamba(nn.Module):

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.H = H
        self.W = W
        self.norm = nn.LayerNorm(dim)
        self.ca = CALayer(dim, 16)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        self.conv1 = conv(dim * 2, dim, 1)

    def forward(self, input):
        skip = input
        ca_output = self.ca(input)
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm(input)
        output = self.block(input)
        output = rearrange(output, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output = torch.cat((output,ca_output), dim=1)
        return skip + self.conv1(output)

class DTransformer(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(DTransformer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
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

class EncoderMambaBlock(nn.Module):
    """mamba encoder block"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.norm1 = nn.LayerNorm(dim)
        #self.mamba = CAMamba(dim, H, W)
        #self.mamba = CDMamba(dim, H, W)
        #self.mamba = SingleMambaBlock(dim, H, W)
        self.mamba = DMamba(dim, H, W)
        self.norm2 = nn.LayerNorm(dim)
        self.trans = DTransformer(dim, 4, False)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output = self.mamba(input)
        input = output + skip

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm2(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output = self.trans(input)
        output = output + skip

        return output

class DWTMambaBlock(nn.Module):
    """Discrete wavelet transform mamba"""

    def __init__(self, dim, H, W, conv = default_conv):
        super().__init__()
        self.C = dim
        self.H = H // 2
        self.W = W // 2

        self.block_HL = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                              if_devide_out=True, use_norm=True, input_h=H // 2, input_w=W // 2)
        self.block_LH = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                              if_devide_out=True, use_norm=True, input_h=H // 2, input_w=W // 2)
        self.block_HH = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                              if_devide_out=True, use_norm=True, input_h=H // 2, input_w=W // 2)

    def forward(self, input):
        # input: (B, N, C)
        b, c, h, w = input.shape
        skip = input

        xfm = DWTForward(J=1, mode='zero', wave='db1').cuda(device_id0)
        ifm = DWTInverse(mode='zero', wave='db1').cuda(device_id0)

        Yl, Yh = xfm(input)
        LL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        LH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)

        LL[:, :, :, :] = Yl
        HL[:, :, :, :] = Yh[0][:, :, 0, :, :]
        LH[:, :, :, :] = Yh[0][:, :, 1, :, :]
        HH[:, :, :, :] = Yh[0][:, :, 2, :, :]

        input_HL = rearrange(HL, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input_LH = rearrange(LH, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input_HH = rearrange(HH, 'b c h w -> b (h w) c', h=self.H, w=self.W)

        output_HL = self.block_HL(input_HL)
        output_LH = self.block_LH(input_LH)
        output_HH = self.block_HH(input_HH)

        output_HL = rearrange(output_HL, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output_LH = rearrange(output_LH, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output_HH = rearrange(output_HH, 'b (h w) c -> b c h w', h=self.H, w=self.W)

        Yl = LL[:, :, :, :]
        Yh[0][:, :, 0, :, :] = output_HL[:, :, :, :]
        Yh[0][:, :, 1, :, :] = output_LH[:, :, :, :]
        Yh[0][:, :, 2, :, :] = output_HH[:, :, :, :]

        output = ifm((Yl, Yh))

        return output + skip

class CDMamba(nn.Module):
    """ShiftMambaBlock and DWTMambaBlock"""

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.H = H
        self.W = W
        self.ca = CALayer(dim, 16)
        self.DWTMambaBlock = DWTMambaBlock(dim, H, W)
        self.conv1 = conv(dim * 2, dim, 1)

    def forward(self, input):
        skip = input
        output1 = self.ca(input)
        output2 = self.DWTMambaBlock(input)
        output = self.conv1(torch.cat((output1, output2), dim=1))
        return output + skip

class DMamba(nn.Module):
    """ShiftMambaBlock and DWTMambaBlock"""

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.H = H
        self.W = W
        self.DWTMambaBlock = DWTMambaBlock(dim, H, W)

    def forward(self, input):
        skip = input
        output = self.DWTMambaBlock(input)
        return output

class CDMamba2(nn.Module):
    """ShiftMambaBlock and DWTMambaBlock"""

    def __init__(self, dim, H, W, conv=default_conv):
        super().__init__()
        self.H = H
        self.W = W
        self.ca = SingleMambaBlock2(dim, H, W)
        self.DWTMambaBlock = DWTMambaBlock(dim, H, W)
        self.conv1 = conv(dim * 2, dim, 1)

    def forward(self, input):
        skip = input
        output1 = self.ca(input)
        output2 = self.DWTMambaBlock(input)
        output = self.conv1(torch.cat((output1, output2), dim=1))
        return output + skip

class SingleMambaBlock2(nn.Module):
    """Single input Mamba"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = rearrange(input, 'b c h w -> b c (h w)', h=self.H, w=self.W)
        input = self.norm(input)
        output = self.block(input)
        output = rearrange(output, 'b c (h w) -> b c h w', h=self.H, w=self.W)
        # output = self.norm1(output)
        return output + skip

class EncoderMambaBlock2(nn.Module):
    """mamba encoder block"""

    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = CDMamba2(dim, H, W)
        self.norm2 = nn.LayerNorm(dim)
        self.trans = DMamba(dim, H, W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output = self.mamba(input)
        input = output + skip

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=self.H, w=self.W)
        input = self.norm2(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        output = self.trans(input)
        output = output + skip

        return output