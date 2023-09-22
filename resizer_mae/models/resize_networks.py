import torch
import torch.nn as nn
import torch.nn.functional as F

"""
self.target_height = 224
        self.target_width = 224
        
        #Resize_layer
        #(1920, 1080) -> (960, 540)
        #self.layer1 = ResizeBlock(in_channels=3, out_channels=32, kernel_size=7, target_height=960, target_width=540)
        
        #(960, 540) -> (480, 270)
        #self.layer2 = ResizeBlock(in_channels=32, out_channels=16, kernel_size=5, target_height=480, target_width=270)
        
        #(480, 270) -> (224, 224)
        #self.layer3 = ResizeBlock(in_channels=3, out_channels=32, kernel_size=7, target_height=224, target_width=224)
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7,
                                  stride=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.01),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.01)
                      )
        
        self.resblock1 = nn.Sequential(
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.1, inplace=True),
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        self.resblock2 = nn.Sequential(
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.1, inplace=True),
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        
        #back layer
        self.conv_back1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                                      stride=1, bias=False)
        self.bn_back1 = nn.BatchNorm2d(32)
        self.conv_back2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1,
                                      stride=1, bias=False)
"""

class ResizeBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=7, slope=0.1):
        super().__init__()
        self.block = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(slope),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(slope)
                      )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, slope=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class ResizeNetwork(nn.Module):
    def __init__(self, num_res_block, target_height, target_width):
        
        super(ResizeNetwork, self).__init__()
        self.num_res_block = num_res_block
        self.target_height = target_height
        self.target_width = target_width
        
        #Resize_layer
        #(1920, 1080) -> (960, 540)
        #self.layer1 = ResizeBlock(in_channels=3, out_channels=32, kernel_size=7, target_height=960, target_width=540)
        
        #(960, 540) -> (480, 270)
        #self.layer2 = ResizeBlock(in_channels=32, out_channels=16, kernel_size=5, target_height=480, target_width=270)
        
        #(480, 270) -> (224, 224)
        self.layer3 = ResizeBlock(in_channels=3, out_channels=32, kernel_size=7)
        
        resblocks = []
        for i in range(self.num_res_block):
            resblocks.append(ResBlock())
        self.res_block = nn.Sequential(*resblocks)
        
        #back layer
        self.conv_back1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                                      stride=1, bias=False)
        self.bn_back1 = nn.BatchNorm2d(32)
        self.conv_back2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1,
                                      stride=1, bias=False)
    
    def forward(self, x):
        origin_x = F.interpolate(x, size=(self.target_height, self.target_width), mode='bilinear',
                                             align_corners=True)
        
        #x = self.layer1(x)
        #x = F.interpolate(size=(960, 540), mode='bilinear', align_corners=True)
        
        #x = self.layer2(x)
        #x = F.interpolate(size=(480, 270), mode='bilinear', align_corners=True)
        
        x = self.layer3(x)
        x = F.interpolate(x, size=(self.target_height, self.target_width), mode='bilinear',
                             align_corners=True)
        
        resized_x = x
        x = self.res_block(x)
        x = self.conv_back1(x)
        x = self.bn_back1(x) + resized_x
        
        x = self.conv_back2(x) + origin_x
        
        return x