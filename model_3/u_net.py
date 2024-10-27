from model_3.mamba import EncoderMambaBlock
from model_3.transformer import DCTM
from common import *
device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

class mamba_Unet(nn.Module):
    """Mamba Transformer U-net"""

    def __init__(self, n_subs, n_mamba, height, width, conv=default_conv):
        """n_subs:每组的维度
        n_feats:总的维度
        n_mamba:mamba快的个数
        height：图像高度
        width：图像宽度
        """
        super(mamba_Unet, self).__init__()

        self.N = n_mamba
        self.H = height
        self.W = width
        self.step1 = nn.Sequential(
            EncoderMambaBlock(n_subs, height, width),
            conv(n_subs, n_subs, 3),
            nn.LeakyReLU(),
        )
        # self.step2 = nn.Sequential(
        #     Down(n_subs, n_subs * 2),
        #     EncoderMambaBlock(n_subs * 2, height // 2, width // 2),
        #     conv(n_subs * 2, n_subs * 2, 3),
        #     nn.LeakyReLU(),
        # )
        self.step2 = nn.Sequential(
            Down(n_subs, n_subs * 2),
            DCTM(n_subs * 2, 6),
            conv(n_subs * 2, n_subs * 2, 3),
            nn.LeakyReLU(),
        )
        self.step3 = nn.Sequential(
            Down(n_subs * 2, n_subs * 4),
            EncoderMambaBlock(n_subs*4, height // 4, width // 4),
        )
        self.up_1 = Up(n_subs * 4, n_subs * 2)
        # self.step4 = nn.Sequential(
        #     EncoderMambaBlock(n_subs * 2, height // 2, width // 2),
        # )
        self.step4 = nn.Sequential(
            DCTM(n_subs * 2, 6),
        )
        self.up_2 = Up(n_subs * 2, n_subs)
        self.step5 = nn.Sequential(
            EncoderMambaBlock(n_subs, height, width),
            nn.Conv2d(n_subs, n_subs, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        skip = x
        x1 = self.step1(x)
        x2 = self.step2(x1)
        x3 = self.step3(x2)
        x3_up = self.up_1(x3, x2)
        x4 = self.step4(x3_up)
        x4_up = self.up_2(x4, x1)
        x5 = self.step5(x4_up)
        y = x5 + skip

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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        return self.conv(x)


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
