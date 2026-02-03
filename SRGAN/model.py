import numpy as np
import math
import torch
import torch.nn as nn

class ResBlock(nn.Module):
  def __init__(self,in_channels):
    super(ResBlock,self).__init__()
    self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.prelu = nn.PReLU()
    self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size = 3,stride=1,padding=1)
    self.bn2 = nn.BatchNorm2d(in_channels)

  def forward(self,x):
    res = self.conv1(x)
    res = self.bn1(res)
    res = self.prelu(res)
    res = self.conv2(res)
    res = self.bn2(res)
    res = res + x
    return res

class UpsampleBlock(nn.Module):
  def __init__(self, in_channels,scaling_factor):
      super(UpsampleBlock, self).__init__()
      self.conv = nn.Conv2d(in_channels, in_channels * scaling_factor**2, kernel_size=3, stride=1, padding=1)
      self.pixel_shuffle = nn.PixelShuffle(scaling_factor)
      self.prelu = nn.PReLU()


  def forward(self,x):
    x = self.conv(x)
    x = self.pixel_shuffle(x)
    x = self.prelu(x)
    return x

class Generator(nn.Module):
  def __init__(self,scaling_factor,num_of_resblocks):
    super(Generator,self).__init__()
    num_of_upsample_blocks = int(math.log(scaling_factor,2))
    self.initial = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=9,stride=1,padding=4),
        nn.PReLU()
    )
    self.mid = nn.Sequential(
        *[ResBlock(64) for _ in range(num_of_resblocks)],
        nn.Conv2d(64,64,stride=1,kernel_size=3,padding=1),
        nn.BatchNorm2d(64))
    self.upsample = nn.Sequential(
        *[UpsampleBlock(64,2) for _ in range(num_of_upsample_blocks)]
    )
    self.output = nn.Conv2d(64,3,kernel_size=9,stride=1,padding=4)
  def forward(self,x):
    x1 = self.initial(x)
    x2 = self.mid(x1) + x1
    x3 = self.upsample(x2)
    x4 = self.output(x3)
    return torch.tanh(x4)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))