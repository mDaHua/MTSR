import torch
import math
import torch.nn as nn

from common import *
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from Vmamba.Myvmamba import *
from pytorch_wavelets import DWTForward, DWTInverse


class MTSR(nn.Module):

    def __init__(self, n_colors, scale, n_feats, height, width, conv=default_conv):
        super(MTSR, self).__init__()
        self.scale = scale
        self.shallow = nn.Sequential(
            MultiScale_feature(n_colors, conv),
            nn.Conv2d(n_colors, n_feats, 3, padding=1),
        )
        self.body = nn.ModuleList()
        for i in range(4):
            self.body.append(Stage(n_feats, n_feats))
        self.upsample = Upsampler(conv, scale, n_feats)
        self.skip_conv = nn.Conv2d(n_colors, n_feats, 3, padding=1)
        self.tail = nn.Conv2d(n_feats, n_colors, 3, padding=1)

    def forward(self, x, lms):
        skip = x
        shallow_x = self.shallow(x)
        xi = shallow_x
        for i in range(4):
            xi = self.body[i](xi)
        body_x = xi

        up_x = self.upsample(body_x + shallow_x)
        output = self.tail(up_x + self.skip_conv(lms))
        return output


class MultiScale_feature(nn.Module):

    def __init__(self, dim, conv=default_conv):
        super().__init__()
        self.dconv1 = conv(dim, dim, 3, dilation=1)
        self.dconv2 = conv(dim, dim, 3, dilation=3)
        self.dconv3 = conv(dim, dim, 3, dilation=5)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(dim * 3, dim, 1, 1)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
        skip = x
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.dconv2(x))
        x3 = self.act(self.dconv3(x))
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.conv(x4)
        y1 = self.ca(x4)
        y2 = self.sa(x4)
        y = y1 + y2
        return y + skip


class Mamba2(nn.Module):
    """Discrete wavelet transform mamba"""

    def __init__(self, dim, scale):
        super(Mamba2, self).__init__()
        self.C = dim
        self.mamba = SS2D(dim, expand=1, d_state=8)
        self.mamba1 = SS2D(dim, expand=1, d_state=8)

    def forward(self, input):
        skip = input
        input = rearrange(input, 'b c h w -> b h w c')
        output = self.mamba(input)
        output = self.mamba1(output)
        output = rearrange(output, 'b h w c -> b c h w')
        return output


class MTB(nn.Module):
    def __init__(self, dim):
        super(MTB, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba2(dim, 8)
        self.tran = Transformer(dim, 8, False)

    def forward(self, input):
        b, c, h, w = input.shape

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=h, w=w)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=h, w=w)
        output = self.mamba(input)
        output = output + skip

        skip = output
        output = rearrange(output, 'b c h w -> b (h w) c', h=h, w=w)
        output = self.norm2(output)
        output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)
        output = self.tran(output)
        output = output + skip

        return output


class Transformer(nn.Module):

    def __init__(self, dim, num_heads, bias, type='spa'):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.type = type
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, input):
        b, c, h, w = input.shape
        q, k, v = self.qkv(input).chunk(3, dim=1)

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


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mtb = MTB(in_channels)
        #conv = nn.Conv2d(dim, dim, 1, 1)

    def forward(self, input):
        skip = input
        z = self.mtb(input)
        return z


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.ca(attention)
        return attention * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.spa = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.spa(concat)
        attention = self.sigmoid(attention)
        return attention * x

