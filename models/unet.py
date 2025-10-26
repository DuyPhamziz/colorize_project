# models/unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class UNet313(nn.Module):
    def __init__(self, n_bins=313, base_c=32):
        super().__init__()
        self.enc1 = DoubleConv(1, base_c)
        self.enc2 = DoubleConv(base_c, base_c*2)
        self.enc3 = DoubleConv(base_c*2, base_c*4)
        self.enc4 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(base_c*12, base_c*4)
        self.dec3 = DoubleConv(base_c*6, base_c*2)
        self.dec2 = DoubleConv(base_c*3, base_c)
        self.final = nn.Conv2d(base_c, n_bins, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d4 = self.up(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return out
