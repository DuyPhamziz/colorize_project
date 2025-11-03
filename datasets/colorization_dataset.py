import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import IMG_SIZE

class ColorizationDataset(Dataset):
    def __init__(self, gray_dir, color_dir, mask_dir=None, augment=False):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(gray_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.augment = augment

        # Albumentations Transform
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2()
            ], additional_targets={'gray':'image','mask':'image'}, is_check_shapes=True)
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ], additional_targets={'gray':'image','mask':'image'}, is_check_shapes=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Paths
        gray_path = os.path.join(self.gray_dir, self.files[idx])
        color_path = os.path.join(self.color_dir, self.files[idx])

        # Load ảnh
        color = np.array(Image.open(color_path).convert("RGB"))     # (H,W,3)
        gray = np.array(Image.open(gray_path).convert("L"))         # (H,W)
        gray = np.expand_dims(gray, axis=2)                          # (H,W,1)

        # Mask
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.files[idx])
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = np.expand_dims(mask, axis=2)                      # (H,W,1)
        else:
            mask = np.zeros_like(gray)                               # default mask = 0

        # Resize tất cả về IMG_SIZE
        color = cv2.resize(color, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        gray  = cv2.resize(gray,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Kiểm tra shape
        assert color.shape[:2] == gray.shape[:2] == mask.shape[:2], \
            f"Shape mismatch: color={color.shape}, gray={gray.shape}, mask={mask.shape}"

        # Augmentation
        augmented = self.transform(image=color, gray=gray, mask=mask)
        color = augmented['image'].float() / 255.0   # scale 0-1
        gray  = augmented['gray'].float() / 255.0
        mask  = augmented['mask'].float() / 255.0

        return gray, color, mask
