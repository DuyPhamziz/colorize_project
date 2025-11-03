# infer.py
import os
import sys
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Thêm đường dẫn để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet.model import ParallelUNet
from config import IMG_SIZE, OUTPUT_DIR, DATA_DIR

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model
# -----------------------------
gen = ParallelUNet().to(device)
gen_path = os.path.join("checkpoints", "unet", "gen_best.pth")
gen.load_state_dict(torch.load(gen_path, map_location=device))
gen.eval()

# -----------------------------
# Thư mục input/output
# -----------------------------
test_gray_dir = os.path.join(DATA_DIR, "test/gray")
test_color_dir = os.path.join(DATA_DIR, "test/color")  # dùng để tính chỉ số
os.makedirs(os.path.join(OUTPUT_DIR, "colorized"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "vis"), exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, "infer_log.txt")
with open(log_file, "w") as f:
    f.write("filename,PSNR,SSIM,MSE\n")

files = sorted([f for f in os.listdir(test_gray_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])

# -----------------------------
# Infer từng ảnh
# -----------------------------
for file_name in tqdm(files, desc="Infer test images"):
    # Load grayscale
    gray_path = os.path.join(test_gray_dir, file_name)
    gray_img = np.array(Image.open(gray_path).convert("L"))
    gray_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    gray_norm = gray_img / 255.0
    gray_tensor = torch.tensor(gray_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Mask giả
    mask_tensor = torch.zeros_like(gray_tensor).to(device)

    # Infer
    with torch.no_grad():
        fake_tensor = gen(gray_tensor, mask_tensor)

    # Lưu ảnh colorized (tensor 0-1)
    save_image(fake_tensor, os.path.join(OUTPUT_DIR, "colorized", file_name))

    # Chuẩn bị ảnh ghép input | fake | target
    gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    fake_np = np.clip(fake_tensor[0].permute(1,2,0).cpu().numpy(), 0, 1)
    fake_disp = (fake_np * 255).astype(np.uint8)

    # Target (nếu có)
    color_path = os.path.join(test_color_dir, file_name)
    if os.path.exists(color_path):
        color_img = np.array(Image.open(color_path).convert("RGB"))
        color_img = cv2.resize(color_img, (IMG_SIZE, IMG_SIZE))
        color_norm = color_img / 255.0
    else:
        color_img = fake_disp
        color_norm = fake_np

    combined = np.concatenate([gray_3ch, fake_disp, color_img], axis=1)
    vis_path = os.path.join(OUTPUT_DIR, "vis", file_name)
    cv2.imwrite(vis_path, combined)

    # Tính PSNR, SSIM, MSE
    mse_val = np.mean((fake_np - color_norm) ** 2)
    psnr_val = 10 * np.log10(1.0 / (mse_val + 1e-8))
    ssim_val = ssim(fake_np, color_norm, channel_axis=-1, data_range=1.0)

    # Ghi log
    with open(log_file, "a") as f:
        f.write(f"{file_name},{psnr_val:.4f},{ssim_val:.4f},{mse_val:.6f}\n")

print("Infer finished! Log saved at:", log_file)
print("Colorized images saved at:", os.path.join(OUTPUT_DIR, "colorized"))
print("Visualization images saved at:", os.path.join(OUTPUT_DIR, "vis"))
