#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23

@author: Haobo CHEN
"""
import torch.nn as nn

class BkConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BkConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.in1   = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.act1  = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        return self.act1(self.in1(self.conv1(x)))


def _make_nConv(in_channels, depth, unit_channel=32):
    layer1 = BkConv(in_channels, unit_channel*(2**depth))
    layer2 = BkConv(unit_channel*(2**depth), unit_channel*(2**depth))

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channels, depth)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 4:
            out_before_pool = self.ops(x)
            out = out_before_pool
            feature = self.avgpool(out_before_pool)
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
            feature = self.avgpool(out_before_pool)

        return out, feature

class UpTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.ops = _make_nConv(in_channels, depth)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())

    def forward(self, x):
        out_up_conv = self.up_conv(x)
        out = self.ops(out_up_conv)
        feature = self.avgpool(out)
        return out, feature

class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, n_labels, kernel_size=1)

    def forward(self, x):
        out = self.final_conv(x)
        return out

class pUNet(nn.Module):
    def __init__(self, n_class):
        super(pUNet, self).__init__()

        self.down_tr32  = DownTransition(n_class, 0)
        self.down_tr64  = DownTransition(32,      1)
        self.down_tr128 = DownTransition(64,      2)
        self.down_tr256 = DownTransition(128,     3)
        self.down_tr512 = DownTransition(256,     4)

        self.up_tr256 = UpTransition(512, 3)
        self.up_tr128 = UpTransition(256, 2)
        self.up_tr64  = UpTransition(128, 1)
        self.up_tr32  = UpTransition(64,  0)

        self.out = OutputTransition(32, n_class)

    def forward(self,x):
        self.enc32,  self.feat_enc32  = self.down_tr32(x)
        self.enc64,  self.feat_enc64  = self.down_tr64(self.enc32)
        self.enc128, self.feat_enc128 = self.down_tr128(self.enc64)
        self.enc256, self.feat_enc256 = self.down_tr256(self.enc128)
        self.enc512, self.feat_enc512 = self.down_tr512(self.enc256)

        self.dec256, self.feat_dec256 = self.up_tr256(self.enc512)
        self.dec128, self.feat_dec128 = self.up_tr128(self.dec256)
        self.dec64,  self.feat_dec64  = self.up_tr64(self.dec128)
        self.dec32,  self.feat_dec32  = self.up_tr32(self.dec64)

        self.feat_set = [self.feat_enc32, self.feat_enc64, self.feat_enc128, self.feat_enc256,
                         self.feat_dec256, self.feat_dec128, self.feat_dec64, self.feat_dec32]

        return self.out(self.dec32), self.feat_set, self.feat_enc512


