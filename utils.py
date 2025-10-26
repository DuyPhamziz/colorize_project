# utils.py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from config import ANNEAL_T, BINS_PATH

def load_bins(path=BINS_PATH):
    return np.load(path)

def annealed_mean_from_logits(logits, bins, T=ANNEAL_T):
    """
    logits: [B,313,H,W]
    bins: numpy (313,2) in OpenCV LAB ab scale
    returns: ab tensor [B,2,H,W] (float) in same scale as bins
    """
    device = logits.device
    B, C, H, W = logits.shape
    prob = F.softmax(logits / T, dim=1)  # [B,313,H,W]
    prob = prob.permute(0,2,3,1).reshape(-1, C)  # [B*H*W,313]
    centers = torch.from_numpy(bins).to(device).float()  # [313,2]
    ab_flat = prob @ centers  # [B*H*W,2]
    ab = ab_flat.reshape(B, H, W, 2).permute(0,3,1,2)  # [B,2,H,W]
    return ab

def lab_to_rgb_batch(L_tensor, ab_tensor):
    """
    L_tensor: [B,1,H,W] L in [0,1] or [0,100]
    ab_tensor: [B,2,H,W] either in OpenCV ab scale (~0..255) or normalized [-1,1]
    returns: numpy array BxHxWx3 uint8
    """
    L = L_tensor.detach().cpu().numpy()
    ab = ab_tensor.detach().cpu().numpy()
    B, _, H, W = L.shape
    out = []
    for i in range(B):
        Li = L[i,0]
        if Li.max() <= 1.1:
            Li = Li * 100.0
        Abi = ab[i].transpose(1,2,0).astype(np.float32)
        if np.abs(Abi).max() <= 2.0:  # likely normalized
            Abi = Abi * 128.0 + 128.0
        lab = np.zeros((H,W,3), dtype=np.float32)
        lab[:,:,0] = np.clip(Li, 0, 100)
        lab[:,:,1:] = np.clip(Abi, 0, 255)
        lab_uint8 = lab.astype(np.uint8)
        rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
        out.append(rgb)
    return np.stack(out, axis=0)
