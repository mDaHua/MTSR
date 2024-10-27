from common import *
from model_2.conv import  *
from model_2.group import MSAMG
from model_2.mamba import SingleMambaBlock_2

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
        #self.pre = Upsampler(conv, scale, n_feats)
        self.up = PixelShuffle(n_colors, scale)
        self.shallow = Conv3(n_colors, n_feats)
        self.group = MSAMG(n_subs, n_ovls, n_colors, n_feats, n_MTB, height, width)
        self.feature_mamba = SingleMambaBlock_2(n_feats, height, width)
        #self.body = MT_Unet(n_feats, n_MTB, height, width)
        self.mamba_group = conv(n_feats, n_feats, 3)
        self.skip_conv = conv(n_colors, n_feats, 3)
        self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x, lms):
        x_up = self.up(x)
        x_shallow =self.shallow(x_up)
        x_group = self.group(x_shallow)
        x_mamba = self.feature_mamba(x_shallow)
        x_deep = self.mamba_group(x_group+x_mamba)
        #x_body = self.body(x_head)
        output = x_deep + self.skip_conv(x_up)
        output = self.tail(output)
        return output




