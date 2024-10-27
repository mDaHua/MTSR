from triton.interpreter.memory_map import torch

from common import *

class MSAMG(nn.Module):

    def __init__(self, n_subs, n_ovls, n_colors, n_feats, n_mamba, height, width, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.n_colors = n_colors
        self.n_feats = n_feats
        self.N = n_mamba
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

        self.spa_spe = MDCCAB(n_subs, n_subs)
        self.conv1 = conv(n_subs * 3, n_subs, 1)
        self.tail = conv(n_colors, n_feats, 3)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]

            if g == 0:
                x_spa_spe = self.spa_spe(xi)
                xi_pre = xi
                x_pre = x_spa_spe
                x_g = x_spa_spe
            else:
                x_inter = torch.cat((xi_pre, x_pre, xi), dim=1)
                x_inter = self.conv1(x_inter)
                x_spa_spe = self.spa_spe(x_inter)
                xi_pre = xi
                x_pre = x_spa_spe
                x_g = x_spa_spe

            y[:, sta_ind:end_ind, :, :] += x_g
            #累加各通道的累加次数
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.tail(y)
        return y

class MDCCAB(nn.Module):
    """ Multi-Scale Dilation Conv Channels Attention Block"""

    def __init__(self, in_channels, out_channels, conv=default_conv):
        super(MDCCAB, self).__init__()
        self.dconv1 = conv(in_channels, out_channels, 3, dilation=1)
        self.dconv2 = conv(in_channels, out_channels, 3, dilation=3)
        self.dconv3 = conv(in_channels, out_channels, 3, dilation=5)
        self.act = nn.ReLU()
        self.ca = nn.Sequential(
            conv(in_channels, out_channels, 3),
            nn.ReLU(),
            CALayer(in_channels,16)
        )
        self.sa = nn.Sequential(
            conv(in_channels, out_channels, 3),
            nn.ReLU(),
            SpatialAttention(kernel_size=7)
        )
        self.conv5 = conv(in_channels * 3, out_channels, 1)
        #self.conv6 = conv(in_channels * 2, out_channels, 1)
        self.conv6 = conv(in_channels * 3, out_channels, 1)

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.dconv2(x))
        x3 = self.act(self.dconv3(x))
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.conv5(x4)
        x5 = self.ca(x)
        x6 = self.sa(x)
        y = torch.cat((x4, x5, x6), dim=1)
        #y = torch.cat((x4, x5), dim=1)
        y = self.conv6(y)
        return y