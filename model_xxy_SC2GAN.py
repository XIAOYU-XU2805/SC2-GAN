#!/usr/bin/python
#-*-coding:utf-8-*-

from __future__ import print_function

# from tensorflow.contrib.layers import xavier_initializer
import scipy.io as io
# from bnorm import VBN

import os
import numpy as np
from six.moves import xrange
import scipy.io as io
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from utils_xxy import CBAM
from utils_xxy import CondConv2D
from torchvision.transforms import ToTensor

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


# SRGAN generator
class AEGenerator1(nn.Module):
    def __init__(self):
        super(AEGenerator1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.blockCBAM3 = CBAM(64)

        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block_cond1 = CondConv2D(64,64)
        self.blockCBAM5 = CBAM(64)

        self.block6 = ResidualBlock(64)
        self.block7_mit = ResidualBlock(64)
        self.block_cond2 = CondConv2D(64,64)
        self.blockCBAM7 = CBAM(64)
        self.block8_mit = ResidualBlock(64)
        self.block9_mit = ResidualBlock(64)
        self.block_cond3 = CondConv2D(64,64)
        self.blockCBAM9 = CBAM(64)
        self.block10_mit = ResidualBlock(64)
        self.block11_mit = ResidualBlock(64)
        self.block_mit = nn.Sequential(
            CondConv2D(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blockCBAM11 = CBAM(64)
        self.block12_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        self.block13_seg = ResidualBlock(64)
        self.block14_seg = ResidualBlock(64)


        self.block_seg = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block18_seg = nn.Conv2d(64, 8, kernel_size=3, padding=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        blockCBAM3 = self.blockCBAM3(block3)

        block4 = self.block4(blockCBAM3)
        block5 = self.block5(block4)
        block_cond1 = self.block_cond1(block5)
        blockCBAM5 = self.blockCBAM5(block_cond1)

        block6 = self.block6(blockCBAM5)
        block_div = block6 + block1
        block7 = self.block7_mit(block_div)
        block_cond2 = self.block_cond2(block7)
        blockCBAM7 = self.blockCBAM7(block_cond2)

        block8 = self.block8_mit(blockCBAM7)
        block9 = self.block9_mit(block8)
        block_cond3 = self.block_cond3(block9)
        blockCBAM9 = self.blockCBAM9(block_cond3)

        block10 = self.block10_mit(blockCBAM9)
        block11 = self.block11_mit(block10)
        block11_mit = self.block_mit(block11)
        blockCBAM11 = self.blockCBAM11(block11_mit)
        out_mit = self.block12_out(blockCBAM11)

        block13_seg = self.block13_seg(block_div)
        block14_seg = self.block14_seg(block13_seg)

        block17_mix = self.block_seg(block14_seg)
        out_seg = self.block18_seg(block17_mix)

        return out_mit, out_seg

class discriminator1(nn.Module):
    def __init__(self):
        super(discriminator1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).reshape([batch_size, -1])

