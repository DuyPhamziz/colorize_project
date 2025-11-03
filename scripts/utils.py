import torch
import torch.nn.functional as F
import math
from skimage.metrics import structural_similarity as ssim
import numpy as np

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def ssim_metric(img1, img2):
    # img1,img2: tensor 3xHxW, value 0-1
    img1 = img1.permute(1,2,0).cpu().numpy()
    img2 = img2.permute(1,2,0).cpu().numpy()
    return ssim(img1, img2, multichannel=True, data_range=1.0)
