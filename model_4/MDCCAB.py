from common import *

class MDCCAB(nn.Module):
    """ Multi-Scale Dilation Conv Channels Attention Block
    """
    def __init__(self, in_channels, out_channels, conv=default_conv):
        super(MDCCAB, self).__init__()
        self.dconv1 = conv(in_channels, out_channels, 3, dilation=1)
        self.dconv2 = conv(in_channels, out_channels, 3, dilation=3)
        self.dconv3 = conv(in_channels, out_channels, 3, dilation=5)
        self.act = nn.PReLU()
        self.dca = nn.Sequential(
            conv(in_channels, out_channels, 3, dilation=1),
            nn.ReLU(),
            CALayer(in_channels,16)
        )
        self.conv6 = DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.dconv2(x))
        x3 = self.act(self.dconv3(x))
        x4 = x1 + x2 + x3
        x5 = self.dca(x)
        x6 = self.conv6(x4 + x5)
        y = x6 + x
        return y
