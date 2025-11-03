#evaluate.py
import torch
from torch.utils.data import DataLoader
from datasets.colorization_dataset import ColorizationDataset
from models.unet.model import ParallelUNet
from scripts.utils import psnr, ssim_metric
from config import *
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = ParallelUNet().to(device)
checkpoint_path = os.path.join(CHECKPOINT_DIR,"unet","gen_epoch50.pth")
gen.load_state_dict(torch.load(checkpoint_path))
gen.eval()

dataset = ColorizationDataset(os.path.join(DATA_DIR,"test/gray"),
                              os.path.join(DATA_DIR,"test/color"),
                              augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

os.makedirs(LOG_DIR, exist_ok=True)

psnr_total = 0
ssim_total = 0
count = 0

with torch.no_grad():
    for gray, color, mask in tqdm(loader, desc="Evaluating"):
        gray, color = gray.to(device), color.to(device)
        fake = gen(gray, mask)
        psnr_total += psnr(fake[0], color[0])
        ssim_total += ssim_metric(fake[0], color[0])
        count += 1

psnr_avg = psnr_total / count
ssim_avg = ssim_total / count

with open(os.path.join(LOG_DIR,"metrics_test.txt"), "w") as f:
    f.write(f"PSNR,SSIM\n{psnr_avg},{ssim_avg}\n")

print(f"Test PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}")
