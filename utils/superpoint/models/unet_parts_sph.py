"""U-net parts used for SuperPointNet_gauss2.py
"""
# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...omniconv.DeformConv2d_sphe import *


class double_conv_deform(nn.Module):
    '''(DeformConv2d_sphe => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(double_conv_deform, self).__init__()
        self.conv = nn.Sequential(
            DeformConv2d_sphe(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DeformConv2d_sphe(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv_deform(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_deform, self).__init__()
        self.conv = double_conv_deform(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_deform(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_deform, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_deform(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


#class up(nn.Module):
#    def __init__(self, in_ch, out_ch, bilinear=True):
#        super(up, self).__init__()
#
#        #  would be a nice idea if the upsampling could be learned too,
#        #  but my machine do not have enough memory to handle all those weights
#        if bilinear:
#            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#        else:
#            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
#
#        self.conv = double_conv(in_ch, out_ch)
#
#    def forward(self, x1, x2):
#        x1 = self.up(x1)
#        
#        # input is CHW
#        diffY = x2.size()[2] - x1.size()[2]
#        diffX = x2.size()[3] - x1.size()[3]
#
#        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                        diffY // 2, diffY - diffY//2))
#        
#        # for padding issues, see 
#        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#
#        x = torch.cat([x2, x1], dim=1)
#        x = self.conv(x)
#        return x
#
#
#class outconv(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(outconv, self).__init__()
#        self.conv = nn.Conv2d(in_ch, out_ch, 1)
#
#    def forward(self, x):
#        x = self.conv(x)
#        return x
#