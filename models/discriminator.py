# models/discriminator.py
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c, base_c*2, 4, 2, 1),
            nn.BatchNorm2d(base_c*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c*2, base_c*4, 4, 2, 1),
            nn.BatchNorm2d(base_c*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)
