import os
import cv2
import shutil
import random

# -----------------------------
# Cấu hình
# -----------------------------
SRC_COLOR_DIR = "data/color"
DST_ROOT = "dataset"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}  # Tỷ lệ
IMG_EXTS = (".png", ".jpg", ".jpeg")

# -----------------------------
# Tạo folder
# -----------------------------
for split in SPLITS.keys():
    os.makedirs(os.path.join(DST_ROOT, split, "color"), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, split, "gray"), exist_ok=True)

# -----------------------------
# Lấy tất cả ảnh màu
# -----------------------------
all_files = [f for f in sorted(os.listdir(SRC_COLOR_DIR)) if f.lower().endswith(IMG_EXTS)]
random.shuffle(all_files)

n_total = len(all_files)
n_train = int(SPLITS["train"] * n_total)
n_val   = int(SPLITS["val"] * n_total)

# -----------------------------
# Chia file
# -----------------------------
splits_files = {
    "train": all_files[:n_train],
    "val": all_files[n_train:n_train+n_val],
    "test": all_files[n_train+n_val:]
}

# -----------------------------
# Copy ảnh màu và tạo ảnh xám
# -----------------------------
for split, files in splits_files.items():
    for f in files:
        src_path = os.path.join(SRC_COLOR_DIR, f)
        
        # Copy ảnh màu
        dst_color_path = os.path.join(DST_ROOT, split, "color", f)
        shutil.copy(src_path, dst_color_path)
        
        # Tạo ảnh grayscale (L channel)
        img_color = cv2.imread(src_path)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        dst_gray_path = os.path.join(DST_ROOT, split, "gray", f)
        cv2.imwrite(dst_gray_path, img_gray)

print("Dataset chuẩn bị xong!")
