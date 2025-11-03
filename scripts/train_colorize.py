import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.colorization_dataset import ColorizationDataset
from models.unet.model import ParallelUNet
from models.gan.model import Discriminator
from config import *
from scripts.utils import psnr

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Dataset
# ------------------------------
train_set = ColorizationDataset(
    os.path.join(DATA_DIR, "train/gray"),
    os.path.join(DATA_DIR, "train/color"),
    mask_dir=os.path.join(DATA_DIR, "train/mask"),
    augment=True
)
val_set = ColorizationDataset(
    os.path.join(DATA_DIR, "val/gray"),
    os.path.join(DATA_DIR, "val/color"),
    mask_dir=os.path.join(DATA_DIR, "val/mask"),
    augment=False
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# ------------------------------
# Models
# ------------------------------
gen = ParallelUNet().to(device)
disc = Discriminator().to(device)

opt_gen = torch.optim.Adam(gen.parameters(), lr=LR_GEN)
opt_disc = torch.optim.Adam(disc.parameters(), lr=LR_DISC)
scaler = torch.amp.GradScaler(device="cuda")

bce_loss = torch.nn.BCEWithLogitsLoss()

# ------------------------------
# Directories
# ------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR + "/colorized", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/vis", exist_ok=True)
os.makedirs(CHECKPOINT_DIR + "/unet", exist_ok=True)
os.makedirs(CHECKPOINT_DIR + "/gan", exist_ok=True)

# ------------------------------
# CSV headers
# ------------------------------
metrics_file = os.path.join(LOG_DIR, "metrics_log.csv")
if not os.path.exists(metrics_file):
    with open(metrics_file, "w") as f:
        f.write("epoch,PSNR,SSIM,MSE\n")

# ------------------------------
# Training loop
# ------------------------------
best_psnr = 0.0

for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, (gray, color, mask) in enumerate(loop):
        gray, color, mask = gray.to(device), color.to(device), mask.to(device)

        # ----------------------
        # Train Discriminator
        # ----------------------
        with torch.amp.autocast(device_type="cuda"):
            fake = gen(gray, mask)
            real_out = disc(color)
            fake_out = disc(fake.detach())
            d_loss = (bce_loss(real_out, torch.ones_like(real_out)) +
                      bce_loss(fake_out, torch.zeros_like(fake_out))) / 2

        opt_disc.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.step(opt_disc)

        # ----------------------
        # Train Generator
        # ----------------------
        with torch.amp.autocast(device_type="cuda"):
            fake = gen(gray, mask)
            pred_fake = disc(fake)
            g_gan_loss = bce_loss(pred_fake, torch.ones_like(pred_fake))
            g_l1_loss = F.l1_loss(fake, color) * LAMBDA_L1
            g_loss = g_gan_loss + g_l1_loss

        opt_gen.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(opt_gen)
        scaler.update()

        # ----------------------
        # Logging batch images
        # ----------------------
        if i % LOG_INTERVAL == 0:
            save_image(
                torch.cat([gray.repeat(1,3,1,1), fake, color], dim=0),
                os.path.join(OUTPUT_DIR, "vis", f"epoch{epoch}_batch{i}.png")
            )

    # ----------------------
    # Validation per epoch
    # ----------------------
    gen.eval()
    psnr_list = []
    ssim_list = []
    mse_list = []

    with torch.no_grad():
        for gray, color, mask in val_loader:
            gray, color, mask = gray.to(device), color.to(device), mask.to(device)
            fake = gen(gray, mask)

            # Convert to numpy 0-255
            fake_np = (fake[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            color_np = (color[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            # MSE
            mse_val = np.mean((fake_np - color_np) ** 2)
            mse_list.append(mse_val)

            # PSNR
            psnr_val = 10 * np.log10(255.0**2 / (mse_val + 1e-8))
            psnr_list.append(psnr_val)

            # SSIM
            ssim_val = ssim(fake_np, color_np, channel_axis=-1, data_range=255)
            ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)
    gen.train()

    # Logging metrics to CSV
    with open(metrics_file, "a") as f:
        f.write(f"{epoch},{avg_psnr:.4f},{avg_ssim:.4f},{avg_mse:.4f}\n")

    print(f"Epoch {epoch}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, MSE={avg_mse:.4f}")

    # ----------------------
    # Save checkpoints
    # ----------------------
    torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR,"unet",f"gen_epoch{epoch}.pth"))
    torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR,"gan",f"disc_epoch{epoch}.pth"))

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR,"unet","gen_best.pth"))
        torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR,"gan","disc_best.pth"))
        print(f"New best PSNR: {best_psnr:.4f}, checkpoint saved.")

print("Training finished!")
