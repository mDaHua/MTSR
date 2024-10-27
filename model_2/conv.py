import torch
import torch.nn as nn

def Conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.LeakyReLU()
    )