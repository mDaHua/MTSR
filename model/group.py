import torch
from common import *
import pywt
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

class MSAMG(nn.Module):
    """Multi-scale spatial attention Module base on Group"""

    def __init__(self, n_subs, n_ovls, n_colors, n_feats, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.n_feats = n_feats
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.dcb = DCB(n_subs)
        self.dwtca = DWTCA(n_subs, n_feats)
        # self.IG = DCAM(n_subs, n_feats)
        # self.spc = nn.ModuleList()
        # self.middle = nn.ModuleList()
        # for n in range(self.G):
        #     self.spc.append(ResAttentionBlock(conv, n_feats, 1, res_scale=0.1))
        #     self.middle.append(conv(n_feats, n_subs, 1))
        self.tail = conv(n_colors, n_feats, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.dcb(xi)
            xi = self.dwtca(xi)
            # xi = self.spc[g](xi)
            # xi = self.middle[g](xi)
            y[:, sta_ind:end_ind, :, :] += xi
            #累加各通道的累加次数
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.tail(y)
        return y

class DCB(nn.Module):
    """Dilated Conv Block"""

    def __init__(self, n_subs, conv=default_conv):
        super(DCB, self).__init__()
        self.dconv1 = conv(n_subs, n_subs, 3, dilation=1)
        self.dconv2 = conv(n_subs, n_subs, 3, dilation=3)
        self.dconv3 = conv(n_subs, n_subs, 3, dilation=5)
        self.act = nn.PReLU()
        self.conv4 = conv(n_subs, n_subs, 1, dilation=1)

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.dconv2(x))
        x3 = self.act(self.dconv3(x))
        y = x1 + x2 + x3
        y = self.conv4(y)
        return y

class DWTCA(nn.Module):
    """Discrete Wavelet Transform and Channels Attention"""

    def __init__(self, n_subs, n_feats, conv=default_conv):
        super(DWTCA, self).__init__()
        self.dwtb = DWTB(n_subs, n_subs)
        self.ca = ResAttentionBlock(conv, n_subs, 1, res_scale=0.1)
        self.conv1 = conv(n_subs, n_subs, 1)
        self.conv2 = conv(n_subs*2, n_subs, 1)
        self.tail = conv(n_subs, n_subs, 3)
        self.act = nn.ReLU()

    def forward(self, x):
        x_spa1 = self.act(x + self.dwtb(x))
        x_spe1 = self.act(x + self.ca(x))
        x_m = self.conv2(torch.cat((x_spa1, x_spe1), dim=1))
        x_spe2 = self.act(x_spe1 + self.ca(x_m))
        x_e = self.conv2(torch.cat((x_spe2, x_spa1), dim=1))
        x_spa2 = self.act(x_spa1 + self.dwtb(x_e))
        y = self.act(self.tail(x_spa2))

        return y

class DWTB(nn.Module):
    """Discrete Wavelet Transform Block"""

    def __init__(self, in_channels, out_channels):
        super(DWTB, self).__init__()
        self.dwconv_HL = nn.Sequential(
            #DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.dwconv_LH = nn.Sequential(
            #DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.dwconv_HH = nn.Sequential(
            #DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        #coeffs = pywt.wavedec2(x, 'haar', level=1)
        #LL, (HL, LH, HH) = coeffs

        #x_idwt = self.dwconv_HH(x)
        device = torch.device("cuda")

        xfm = DWTForward(J=1, mode='zero', wave='haar').cuda(device_id0)
        ifm = DWTInverse(mode='zero', wave='haar').cuda(device_id0)

        Yl, Yh = xfm(x)

        h00 = torch.zeros(x.shape).float().cuda(device_id0)
        h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
        h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[0][:, :, 0, :, :]
        h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[0][:, :, 1, :, :]
        h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[0][:, :, 2, :, :]

        h11 = self.dwconv_HL(h00)

        Yl = h11[:, :, :Yl.size(2), :Yl.size(3)]
        Yh[0][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2]
        Yh[0][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)]
        Yh[0][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2]

        Yl = Yl.cuda(device_id0)

        # LL = Yl
        # HL = Yh[0][:, :, 0, :, :]
        # LH = Yh[0][:, :, 1, :, :]
        # HH = Yh[0][:, :, 2, :, :]
        # HL = self.dwconv_HL(HL)
        # LH = self.dwconv_LH(LH)
        # HH = self.dwconv_HH(HH)
        # print("x.shape",x.shape)
        # print("HL.shape",HL.shape)
        #
        # Yl = LL.cuda(device_id0)
        # Yh[0][:, :, 0, :, :] = HL
        # Yh[0][:, :, 1, :, :] = LH
        # Yh[0][:, :, 2, :, :] = HH


        x_idwt = ifm((Yl, Yh))
        #x_idwt = pywt.waverec2((LL,(HL, LH, HH)),'haar').cuda(device_id0)
        return x_idwt
