from triton.interpreter.memory_map import torch

from common import *
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

class MTSR(nn.Module):

    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_MTB, height, width, conv=default_conv):
        super(MTSR, self).__init__()
        self.N = n_MTB
        self.trans = nn.ModuleList()
        self.mamba = nn.ModuleList()
        self.mtblock = nn.ModuleList()
        self.shallow = nn.Sequential(
            conv(n_colors, n_colors, 3),
            conv(n_colors, n_feats, 1),
            nn.ReLU()
        )
        for i in range(self.N):
            self.trans.append(TransBlock(n_feats, height, width))
            self.mamba.append(MambaBlock(n_feats, height, width))
            self.mtblock.append(MTBlock(n_feats, height, width))
        self.conv = nn.Conv2d(n_feats, n_colors, 1)
        self.up = Upsampler(conv, scale, n_colors)
        self.skip_conv = conv(n_colors, n_colors, 3)
        self.tail = conv(n_colors, n_colors, 3)

    def forward(self, input, lms):
        input_shallow =self.shallow(input)

        input_t_1 = self.trans[0](input_shallow)
        input_m_1 = self.mamba[0](input_shallow)
        input_mt_1 = self.mtblock[0](input_t_1, input_m_1)

        input_t_2 = self.trans[0](input_t_1)
        input_m_2 = self.mamba[0](input_m_1 + input_mt_1)
        input_mt_2 = self.mtblock[0](input_t_2, input_m_2)

        input_t_3 = self.trans[0](input_t_2)
        input_m_3 = self.mamba[0](input_m_2 + input_mt_2)
        input_mt_3 = self.mtblock[0](input_t_3, input_m_3)

        input_t_4 = self.trans[0](input_t_3)
        input_m_4 = self.mamba[0](input_m_3 + input_mt_3)
        input_mt_4 = self.mtblock[0](input_t_4, input_m_4)

        output = self.conv(input_mt_4 + input_shallow)
        output = self.up(output)
        output = output + self.skip_conv(lms)
        output = self.tail(output)
        return output

class Transformer(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        #q = self.deformconv(x)
        q , k, v = self.qkv(x).chunk(3, dim=1)

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

class SingleMambaBlock(nn.Module):
    """Single input Mamba"""

    def __init__(self, dim, H, W):
        super(SingleMambaBlock, self).__init__()
        self.H = H
        self.W = W
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        output = self.block(input)
        return output

class DWT(nn.Module):
    """Discrete wavelet transform mamba"""

    def __init__(self, dim, H, W, conv = default_conv):
        super(DWT, self).__init__()
        self.C = dim
        self.H = H // 2
        self.W = W // 2

        self.block_HL = conv(dim, dim, 3)
        self.block_LH = conv(dim, dim, 3)
        self.block_HH = conv(dim, dim, 3)
        self.act = nn.LeakyReLU()

    def forward(self, input):
        # input: (B, N, C)
        b, c, h, w = input.shape

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

        output_HL = self.act(self.block_HL(HL))
        output_LH = self.act(self.block_LH(LH))
        output_HH = self.act(self.block_HH(HH))

        Yl = LL[:, :, :, :]
        Yh[0][:, :, 0, :, :] = output_HL[:, :, :, :]
        Yh[0][:, :, 1, :, :] = output_LH[:, :, :, :]
        Yh[0][:, :, 2, :, :] = output_HH[:, :, :, :]

        output = ifm((Yl, Yh))

        return output

class TransBlock(nn.Module):

    def __init__(self, dim, H, W, drop_path=0.0):
        super(TransBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.trans = Transformer(dim, 4, False)
        self.dwt = DWT(dim, H, W)

    def forward(self, input):
        B, C, H, W = input.shape

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', head=self.num_heads, h=H, w=W)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', head=self.num_heads, h=H, w=W)
        output = self.trans(input)
        output = skip + self.drop_path(output)

        skip = output
        output = rearrange(output, 'b c h w -> b (h w) c', head=self.num_heads, h=H, w=W)
        output = self.norm2(output)
        output = rearrange(output, 'b (h w) c -> b c h w', head=self.num_heads, h=H, w=W)
        output = skip + self.drop_path(self.dwt(output))

        return output

class MambaBlock(nn.Module):

    def __init__(self, dim, H, W, drop_path=0.0):
        super(MambaBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mamba = SingleMambaBlock(dim, H, W)
        self.dwt = DWT(dim, H, W)

    def forward(self, input):
        B, C, H, W = input.shape

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', head=self.num_heads, h=H, w=W)
        input = self.norm1(input)
        output = self.mamba(input)
        output = rearrange(output, 'b (h w) c -> b c h w', head=self.num_heads, h=H, w=W)
        output = skip + self.drop_path(output)

        skip = output
        output = rearrange(output, 'b c h w -> b (h w) c', head=self.num_heads, h=H, w=W)
        output = self.norm2(output)
        output = rearrange(output, 'b (h w) c -> b c h w', head=self.num_heads, h=H, w=W)
        output = skip + self.drop_path(self.dwt(output))

        return output

class MTBlock(nn.Module):

    def __init__(self, dim, H, W):
        super(MTBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x, y):
        input1 = x
        input2 = y
        input1 = self.conv1(input1)
        input2 = self.conv2(input2)
        input = torch.cat([input1, input2], dim=1)
        avg_input = torch.mean(input, dim=1, keepdim=True)
        max_input, _ = torch.max(input, dim=1, keepdim=True)
        agg = torch.cat([avg_input, max_input], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        output = input1 * sig[:, 0, :, :].unsqueeze(1) + input2 * sig[:, 1, :, :].unsqueeze(1)
        output = self.conv(output)
        return (x + y) * output