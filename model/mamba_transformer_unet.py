from common import *
from model.mamba import SingleMambaBlock
from model.transformer import DCTM
from utils import M_or_T

class MT_Unet(nn.Module):
    """Mamba Transformer U-net"""
    def __init__(self, n_feats, n_MTB, Height, Width):
        super(MT_Unet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.N = n_MTB
        self.H = Height
        self.W = Width

        # dimension for each stage
        dim0 = n_feats
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # main body
        self.stage0 = Stage(dim0, dim1, self.H, self.W, 2, 0, self.N, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, self.H // 2, self.W // 2, 2, 1, self.N, sample_mode='down')
        self.stage2 = Stage(dim2, dim2, self.H // 4, self.W // 4, 2, 2, self.N, sample_mode='flat')
        self.stage3 = Stage(dim2, dim1, self.H // 4, self.W // 4, 2, 3, self.N, sample_mode='up')
        self.stage4 = Stage(dim1, dim0, self.H // 2, self.W // 2, 2, 4, self.N, sample_mode='up')
        self.stage5 = Stage(dim0, dim0, self.H, self.W, 2, 5, self.N, sample_mode='flat')

        self.tail = nn.Conv2d(dim0, dim0, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        hs0 = self.head(x)

        hs1, hs_skip1 = self.stage0(hs0)
        hs2, hs_skip2 = self.stage1(hs1)
        hs3, _ = self.stage2(hs2)
        hs4 = self.stage3(hs3, hs_skip2)
        hs5 = self.stage4(hs4, hs_skip1)
        hs, _ = self.stage5(hs5)

        y = x + hs0
        y = self.tail(y)
        return y

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale, upsample='default'):
        super().__init__()
        if upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'bicubic':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'pixelshuffle':
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale, downsample='default'):
        super().__init__()
        if downsample == 'maxpooling':
            self.down = nn.Sequential(
                #最大池化，默认情况下步幅与窗口大小相同
                nn.MaxPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.down(x)


class Flat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.flat = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.flat(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, scale, stage_num, stage_all, sample_mode='down'):
        super().__init__()
        if M_or_T(stage_num, stage_all) == 0:
            self.body = SingleMambaBlock(in_channels, height, width)
        else:
            self.body = DCTM(in_channels, 8, False)
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)
        elif sample_mode == "flat":
            self.sample = Flat(in_channels, out_channels)

    def forward(self, hs, hs_pre=None):
        hs = self.body(hs)
        if hs_pre is None:
            hs_skip = hs
            hs = self.sample(hs)
            return hs, hs_skip
        else:
            hs = self.sample(hs, hs_pre)
            return hs