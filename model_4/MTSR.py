from common import *
from model_4.group import MSAMG

class MTSR(nn.Module):
    """Mamba Transformer super-Resolution (MSTR)
    n_subs:Dim after grouping
    n_ovls:Overlap dim for each group
    n_colors:Hyperspectral image dim
    n_feats:Dim after shallow feature extraction
    default_conv:The default is 1*1 convolution
    n_MTB:Number of MTB blocks
    height，width:Hyperspectral image height，width
    """
    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_MTB, height, width, conv=default_conv):
        super(MTSR, self).__init__()
        self.up = Upsampler(conv, scale, n_colors)
        self.shallow = conv(n_colors, n_feats, 1)
        self.group = MSAMG(n_subs, n_ovls, n_colors, n_feats, n_MTB, height, width)
        self.skip_conv = conv(n_colors, n_feats, 1)
        self.tail = conv(n_feats, n_colors, 1)

    def forward(self, x, lms):
        x_up = self.up(x)
        x_shallow =self.shallow(x_up)
        x_group = self.group(x_shallow)
        output = x_group + self.skip_conv(x_up)
        output = self.tail(output)

        return output




