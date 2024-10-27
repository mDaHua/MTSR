from triton.interpreter.memory_map import torch

from common import *
from model_5.group2 import MSAMG
from model_5.u_net import mamba_Unet
from model_5.mamba import SingleMambaBlock, CDMamba, CAMamba


class MTSR(nn.Module):

    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_MTB, height, width, conv=default_conv):
        super(MTSR, self).__init__()

        self.shallow = conv(n_colors, n_colors, 3)
        self.group = MSAMG(n_subs, n_ovls, n_colors, n_feats, n_MTB, height, width)
        self.u_mamba = mamba_Unet(n_feats, n_MTB, height, width)
        self.conv1 = conv(n_colors, n_feats, 1)
        #self.branch_mamba = CDMamba(n_feats, height, width)
        self.branch_mamba = CAMamba(n_feats, height, width)
        self.conv2 = conv(n_feats * 2, n_feats, 1)
        self.up = Upsampler(conv, scale, n_feats)
        self.skip_conv = conv(n_colors, n_feats, 3)
        self.tail = nn.Sequential(
            conv(n_feats, n_feats, 3),
            conv(n_feats, n_colors, 3),
        )

    def forward(self, x, lms):
        x_shallow =self.shallow(x)
        x_group = self.group(x_shallow)
        x_u_mamba = self.u_mamba(x_group)
        x_up_dim = self.conv1(x_shallow)
        x_b_mamba = self.branch_mamba(x_up_dim)
        x_body = self.conv2(torch.cat((x_u_mamba, x_b_mamba), dim=1))
        x_up = self.up(x_body)
        x_skip = self.skip_conv(lms)
        output = x_up + x_skip
        output = self.tail(output)
        return output




