# datasets.py
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import GRAY_DIR, COLOR_DIR, IMG_SIZE, BINS_PATH

transform_L = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])

def read_list(list_path):
    with open(list_path, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

class ColorizationDataset(Dataset):
    """
    If BINS_PATH exists -> returns (L_tensor, target_idxs, rel_path)
      target_idxs: HxW long tensor of bin indices
    Else -> returns (L_tensor, ab_tensor_norm, rel_path)
      ab_tensor_norm: 2xHxW floats in [-1,1]
    """
    def __init__(self, split_file, gray_root=GRAY_DIR, color_root=COLOR_DIR, bins_path=BINS_PATH):
        self.gray_root = Path(gray_root)
        self.color_root = Path(color_root)
        self.rel_paths = read_list(split_file)
        if len(self.rel_paths) == 0:
            raise RuntimeError('Empty split file: ' + split_file)
        self.bins_path = Path(bins_path)
        if self.bins_path.exists():
            self.bins = np.load(self.bins_path)
        else:
            self.bins = None

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = Path(self.rel_paths[idx])
        pL = self.gray_root / rel
        pRGB = self.color_root / rel
        if not pL.exists() or not pRGB.exists():
            raise FileNotFoundError(f'{pL} or {pRGB} missing (expected matching filenames in gray/color)')

        L_im = Image.open(pL).convert('L').resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)
        RGB_im = Image.open(pRGB).convert('RGB').resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)
        L = transform_L(L_im)  # [1,H,W]

        RGB = np.array(RGB_im)
        lab = cv2.cvtColor(RGB, cv2.COLOR_RGB2LAB).astype(np.float32)
        ab = lab[:,:,1:3].copy()  # H,W,2 in OpenCV scale (~0..255)

        if self.bins is None:
            ab_norm = (ab - 128.0) / 128.0
            ab_t = torch.from_numpy(ab_norm.transpose(2,0,1)).float()
            return L, ab_t, str(rel)
        else:
            H,W,_ = ab.shape
            flat_ab = ab.reshape(-1,2)
            d = np.sum((flat_ab[:,None,:] - self.bins[None,:,:])**2, axis=2)
            idxs = np.argmin(d, axis=1).astype(np.int64).reshape(H,W)
            target = torch.from_numpy(idxs).long()
            return L, target, str(rel)
