import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.down(x))

class ParallelUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, features=[32,64,128]):
        super().__init__()
        self.inc = DoubleConv(in_ch, features[0])
        self.down1 = UNetBlock(features[0], features[1])
        self.down2 = UNetBlock(features[1], features[2])
        self.up1 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.up_conv1 = DoubleConv(features[2], features[1])
        self.up2 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.up_conv2 = DoubleConv(features[1], features[0])
        self.outc = nn.Conv2d(features[0], out_ch, 1)
        
    def forward(self, x, mask=None):
        # mask can be concatenated if needed
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv2(x)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x
