import torch
from torch import nn as nn

from learning.modules.blocks import ResBlockStrided, ResBlock


class ResNet101(torch.nn.Module):
    def __init__(self, channels, down_pad=True):
        super(ResNet101, self).__init__()

        down_padding = 0
        if down_pad:
            down_padding = 1

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1) #
        self.block1 = ResBlockStrided(128, stride=1, down_padding=down_padding)
        self.block15 = ResBlock(128)
        self.block2 = ResBlockStrided(256, stride=1, down_padding=down_padding)
        self.block25 = ResBlock(256)
        self.block3 = ResBlockStrided(256, stride=2, down_padding=down_padding)
        self.block35 = ResBlock(256)
        self.block4 = ResBlockStrided(512, stride=2, down_padding=down_padding)
        self.block45 = ResBlock(512)
        self.block5 = ResBlockStrided(512, stride=2, down_padding=down_padding)
        self.block55 = ResBlock(512)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.res_norm = nn.InstanceNorm2d(512)

    def init_weights(self):
        self.block1.init_weights()
        self.block2.init_weights()
        self.block3.init_weights()
        self.block4.init_weights()
        self.block5.init_weights()

        self.block15.init_weights()
        self.block25.init_weights()
        self.block35.init_weights()
        self.block45.init_weights()
        self.block55.init_weights()


        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        x = self.block15(x)
        x = self.block2(x)
        x = self.block25(x)
        x = self.block3(x)
        x = self.block35(x)
        f_w = x
        x = self.block4(x)
        x = self.block45(x)
        x = self.block5(x)
        x = self.block55(x)
        x = self.res_norm(x)
        return x , f_w