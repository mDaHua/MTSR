from common import *
from model.group import MSAMG
from model.mamba_transformer_unet import MT_Unet

class MTSR(nn.Module):
    """Mamba Transformer super-Resolution (MSTR)
    """
    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_MTB, height, width, conv=default_conv):
        super(MTSR, self).__init__()
        #self.pre = Upsampler(conv, scale, n_feats)
        self.pre = PixelShuffle(n_colors, scale)
        self.head = MSAMG(n_subs, n_ovls, n_colors, n_feats)
        self.body = MT_Unet(n_feats, n_MTB, height, width)
        self.skip_conv = conv(n_colors, n_feats, 1)
        self.tail = conv(n_feats, n_colors, 1)

    def forward(self, x, lms):
        x_up = self.pre(x)
        x_head = self.head(x_up)
        x_body = self.body(x_head)
        output = x_body + self.skip_conv(lms)
        output = self.tail(output)
        return output




