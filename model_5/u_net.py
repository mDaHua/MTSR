from model_5.mamba import EncoderMambaBlock
from common import *

device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

# class mamba_Unet(nn.Module):
#     """Mamba Transformer U-net"""
#
#     def __init__(self, n_feats, n_mamba, height, width, conv=default_conv):
#         super(mamba_Unet, self).__init__()
#
#         self.N = n_mamba
#         self.H = height
#         self.W = width
#         self.step1 = nn.Sequential(
#             EncoderMambaBlock(n_feats, height, width),
#             conv(n_feats, n_feats, 3),
#             conv(n_feats, n_feats, 3),
#             nn.ReLU()
#         )
#         self.step2 = nn.Sequential(
#             EncoderMambaBlock(n_feats, height, width),
#             conv(n_feats, n_feats, 3),
#             conv(n_feats, n_feats, 3),
#             nn.ReLU()
#         )
#         self.step3 = nn.Sequential(
#             EncoderMambaBlock(n_feats, height, width),
#             conv(n_feats, n_feats, 3),
#             conv(n_feats, n_feats, 3),
#             nn.ReLU()
#         )
#         self.step4 = nn.Sequential(
#             EncoderMambaBlock(n_feats, height, width),
#             conv(n_feats, n_feats, 3),
#             conv(n_feats, n_feats, 3),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         skip = x
#         x1 = self.step1(x)
#         x2 = self.step2(x1)
#         x3 = self.step3(x2)
#         x4 = self.step4(x3)
#         y = x4 + skip
#
#         return y

class mamba_Unet(nn.Module):
    """Mamba Transformer U-net"""

    def __init__(self, n_feats, n_mamba, height, width, conv=default_conv):
        super(mamba_Unet, self).__init__()

        self.N = n_mamba
        self.H = height
        self.W = width
        self.step1 = nn.Sequential(
            EncoderMambaBlock(n_feats, height, width),
            conv(n_feats, n_feats, 3),
        )
        self.step2 = nn.Sequential(
            Down(n_feats, n_feats * 2),
            EncoderMambaBlock(n_feats * 2, height // 2, width // 2),
            conv(n_feats * 2, n_feats * 2, 3),
            nn.ReLU(),
        )
        self.step3 = nn.Sequential(
            EncoderMambaBlock(n_feats * 2, height // 2, width // 2),
            conv(n_feats * 2, n_feats * 2, 3),
        )
        self.up = Up(n_feats * 2, n_feats)
        self.step4 = nn.Sequential(
            EncoderMambaBlock(n_feats, height, width),
            conv(n_feats, n_feats, 3),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        skip = x
        x1 = self.step1(x)
        x2 = self.step2(x1)
        x3 = self.step3(x2)
        x3_up = self.up(x3)
        x4 = self.step4(x3_up)
        y = x4 + skip
        return y


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # print("x的形状：" , x.shape)
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, upsample='default'):
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
                PixelShuffle(in_channels, scale),
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

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, downsample='default'):
        super().__init__()
        if downsample == 'maxpooling':
            self.down = nn.Sequential(
                # 最大池化，默认情况下步幅与窗口大小相同
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

    def forward(self, x1):
        return self.down(x1)
