import torch
import torch.nn as nn 

from learning.modules.blocks import ResBlockStrided, ResBlock

class MapsEmbedding(nn.Module):
    def __init__(self,channels, down_pad=True):
        super().__init__()

        if down_pad:
            down_padding = 1
        else:
            down_padding = 0

        self.conv1 = nn.Conv2d(32, channels, 3, stride=1, padding=down_padding)
        self.block4 = ResBlock(channels)
        self.block45 = ResBlockStrided(channels, stride=2, down_padding = down_padding)
        self.block5 = ResBlock(channels)
        self.block55 = ResBlockStrided(channels, stride=2, down_padding=down_padding)
        self.res_norm = nn.InstanceNorm2d(channels)
        
    def init_weights(self):
        self.block4.init_weights()
        self.block45.init_weights()
        self.block5.init_weights()
        self.block55.init_weights()
        
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
            
    def forward(self, maps):
        # print(f"MAPS: {maps.size()}")
        x = self.conv1(maps)
        map32 = x
        # print(f"X AFTER CONV1: {x.size()}")
        x = self.block4(x)
        # print(f"X AFTER BLOCK4: {x.size()}")
        x = self.block45(x)
        # print(f"X AFTER BLOCK45: {x.size()}")
        map16 = x
        x = self.block5(x)
        # print(f"X AFTER BLOCK5: {x.size()}")
        x = self.block55(x)
        # print(f"X AFTER BLOCK55: {x.size()}")
        x = self.res_norm(x)
        # print(x.size())
        return x, map16, map32