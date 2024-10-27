import torch
from common import *
from model_2.u_net import mamba_Unet
import pywt
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
from model_2.mamba import SingleMambaBlock
device_id0 = 'cuda:0'
torch.autograd.set_detect_anomaly(True)

class MSAMG(nn.Module):
    """Multi-scale spatial attention Module base on Group
    n_subs:Dim after grouping
    n_ovls:Overlap dim for each group
    n_feats:Dim after shallow feature extraction
    default_conv:The default is 1*1 convolution
    n_mamba:Number of mambas
    """

    def __init__(self, n_subs, n_ovls, n_colors, n_feats, n_mamba, height, width, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_feats - n_ovls) / (n_subs - n_ovls))
        self.n_feats = n_feats
        self.N = n_mamba
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_feats:
                end_ind = n_feats
                sta_ind = n_feats - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.mamba = mamba_Unet(n_subs, n_mamba, height, width)
        # self.mamba = nn.ModuleList()
        # for i in range(self.N):
        #     self.mamba.append(SingleMambaBlock(n_subs, height, width))
        self.conv_1 = conv(n_subs, n_subs, 1)
        self.tail = conv(n_feats, n_feats, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            x_m = xi
            x_m = self.mamba(x_m)
            # for i in range(self.N):
            #     x_m = self.mamba[i](x_m)
            x_r = x_m + xi
            xi = self.conv_1(x_r)
            y[:, sta_ind:end_ind, :, :] += xi
            #累加各通道的累加次数
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.tail(y)
        return y