from model_2.mamba import SingleMambaBlock, SingleMambaBlock_2
from utils import M_or_T
from common import *
from pytorch_wavelets import DWTForward, DWTInverse
device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

class mamba_Unet(nn.Module):
    """Mamba Transformer U-net"""

    def __init__(self, n_subs, n_mamba, height, width):
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
        self.mamba1 = nn.Sequential(
            SingleMambaBlock_2(n_subs, height, width),
            nn.Conv2d(n_subs, n_subs, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.mamba2 = nn.Sequential(
            SingleMambaBlock(n_subs, height // 2, width // 2),
            nn.Conv2d(n_subs, n_subs, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.mamba3 = nn.Sequential(
            SingleMambaBlock(n_subs, height // 2, width // 2),
            nn.Conv2d(n_subs, n_subs, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.mamba4 = nn.Sequential(
            SingleMambaBlock_2(n_subs, height, width),
            nn.Conv2d(n_subs, n_subs, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.tail = nn.Conv2d(n_subs, n_subs, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_1 = self.mamba1(x)

        xfm = DWTForward(J=1, mode='zero', wave='haar').cuda(device_id0)
        ifm = DWTInverse(mode='zero', wave='haar').cuda(device_id0)

        Yl, Yh = xfm(x_1)
        LL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        LH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)

        LL[:, :, :, :] = Yl
        HL[:, :, :, :] = Yh[0][:, :, 0, :, :]
        LH[:, :, :, :] = Yh[0][:, :, 1, :, :]
        HH[:, :, :, :] = Yh[0][:, :, 2, :, :]

        HL = self.mamba2(HL)
        LH = self.mamba2(LH)
        HH = self.mamba2(HH)

        HL_2 = self.mamba3(HL)
        LH_2 = self.mamba3(LH)
        HH_2 = self.mamba3(HH)

        Yl = LL[:, :, :, :]
        Yh[0][:, :, 0, :, :] = HL_2[:, :, :, :]
        Yh[0][:, :, 1, :, :] = LH_2[:, :, :, :]
        Yh[0][:, :, 2, :, :] = HH_2[:, :, :, :]

        x_idwt = ifm((Yl, Yh))
        y = self.mamba4(x_idwt)

        return y