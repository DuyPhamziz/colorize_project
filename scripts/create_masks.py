# scripts/create_masks.py
import os
import random
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, IMG_SIZE

# ----------------------
# Cấu hình mask
# ----------------------
SPLITS = ["train", "val", "test"]
NUM_BOX_MIN = 1  # số ô trắng tối thiểu
NUM_BOX_MAX = 3  # số ô trắng tối đa
MIN_BOX_SIZE = 16
MAX_BOX_SIZE = 64

# ----------------------
# Hàm tạo mask ngẫu nhiên
# ----------------------
def create_mask(img_size=IMG_SIZE):
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    num_boxes = random.randint(NUM_BOX_MIN, NUM_BOX_MAX)
    for _ in range(num_boxes):
        w = random.randint(MIN_BOX_SIZE, MAX_BOX_SIZE)
        h = random.randint(MIN_BOX_SIZE, MAX_BOX_SIZE)
        x = random.randint(0, img_size - w)
        y = random.randint(0, img_size - h)
        mask[y:y+h, x:x+w] = 255
    return mask

# ----------------------
# Tạo mask cho từng split
# ----------------------
for split in SPLITS:
    gray_dir = os.path.join(DATA_DIR, split, "gray")
    mask_dir = os.path.join(DATA_DIR, split, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(gray_dir) if f.lower().endswith((".jpg", ".png"))])

    for f in tqdm(files, desc=f"Creating masks for {split}"):
        mask = create_mask()
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(mask_dir, f))

print("✅ Done! Masks created for train/val/test.")
