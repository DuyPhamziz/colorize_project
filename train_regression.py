# train_regression.py
"""
Train UNet regression model (predict ab channels directly).
Usage:
  python train_regression.py         # train from scratch
  python train_regression.py --resume checkpoints/unet_reg_epoch10.pth
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio
import numpy as np
from tqdm import tqdm

from config import DEVICE, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_DIR, IMG_SIZE
from datasets import ColorizationDataset
# We will reuse model building blocks but create a small UNet regressor here
from models.unet import DoubleConv

from utils import lab_to_rgb_batch
from config import COLOR_DIR
# -------------------------
# UNet regression (2-channel output)
# -------------------------
class UNetReg(nn.Module):
    def __init__(self, out_ch=2, base_c=32):
        super().__init__()
        self.enc1 = DoubleConv(1, base_c)
        self.enc2 = DoubleConv(base_c, base_c*2)
        self.enc3 = DoubleConv(base_c*2, base_c*4)
        self.enc4 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(base_c*12, base_c*4)
        self.dec3 = DoubleConv(base_c*6, base_c*2)
        self.dec2 = DoubleConv(base_c*3, base_c)
        self.final = nn.Sequential(
            nn.Conv2d(base_c, out_ch, kernel_size=1),
            nn.Tanh()  # constrain to [-1,1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.up(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.final(d2)
        return out

# -------------------------
# Training loop
# -------------------------
def train(args):
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # required split files
    train_split = 'splits/train.txt'
    val_split = 'splits/val.txt'
    if not os.path.exists(train_split) or not os.path.exists(val_split):
        raise RuntimeError("Please run prepare_splits.py to create splits/train.txt and splits/val.txt")

    train_ds = ColorizationDataset(train_split)  # in regression mode dataset returns (L, ab_norm, rel)
    val_ds = ColorizationDataset(val_split)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, BATCH_SIZE//2), shuffle=False, num_workers=2, pin_memory=True)

    model = UNetReg(out_ch=2, base_c=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=DEVICE)
        sd = ck.get('model', ck) if isinstance(ck, dict) else ck
        model.load_state_dict(sd)
        if 'optimizer' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer'])
            except Exception:
                print('Could not load optimizer state (continue with fresh optimizer)')
        start_epoch = ck.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} -> start_epoch = {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        iters = 0
        pbar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
        for L, ab, rel in pbar:
            # dataset returns ab normalized in [-1,1] for regression fallback
            L = L.to(DEVICE)           # [B,1,H,W]
            ab = ab.to(DEVICE)         # [B,2,H,W] with values in approx [-1,1]

            optimizer.zero_grad()
            pred_ab = model(L)         # output in [-1,1] due to Tanh
            loss = criterion(pred_ab, ab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            iters += 1
            pbar.set_postfix({'loss': f'{running_loss/iters:.4f}'})

        avg_train_loss = running_loss / max(1, iters)
        print(f"Epoch {epoch}/{EPOCHS} - Train L1 loss: {avg_train_loss:.6f}")

        # validation quick pass and save sample
        model.eval()
        with torch.no_grad():
            Lval, abval, rels = next(iter(val_loader))
            Lval = Lval.to(DEVICE)
            pred_ab_val = model(Lval).clamp(-1.0, 1.0)
            # convert to RGB for save using lab_to_rgb_batch (it handles normalized ab)
            rgb_batch = lab_to_rgb_batch(Lval, pred_ab_val)  # returns uint8 numpy
            sample_path = os.path.join(str(CHECKPOINT_DIR), f'reg_epoch{epoch}_sample.png')
            imageio.imwrite(sample_path, rgb_batch[0])
            print('Saved sample', sample_path)

        # save checkpoint
        ck = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        ckpt_path = os.path.join(str(CHECKPOINT_DIR), f'unet_reg_epoch{epoch}.pth')
        torch.save(ck, ckpt_path)
        print('Saved checkpoint', ckpt_path)

    print('Training finished.')

# -------------------------
# Optional: evaluation on validation or test set (batch inference)
# -------------------------
def eval_on_split(checkpoint, split_file='splits/test.txt', out_csv='regression_results.csv'):
    """
    Run regression model on split_file and compute PSNR/SSIM vs GT images.
    """
    import csv
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    from PIL import Image

    bins_exist = False  # not used in regression
    model = UNetReg(out_ch=2).to(DEVICE)
    ck = torch.load(checkpoint, map_location=DEVICE)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.eval()

    ds = ColorizationDataset(split_file)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    rows = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    for L, ab_gt, rel in tqdm(loader, desc='Eval regression'):
        L = L.to(DEVICE)
        with torch.no_grad():
            pred_ab = model(L).clamp(-1.0, 1.0)
            rgb_pred = lab_to_rgb_batch(L, pred_ab)[0].astype(np.float32) / 255.0

        # load GT resized (ensure same resize logic as dataset)
        rel_path = rel if isinstance(rel, str) else rel[0]
        gt_path = os.path.join(str(COLOR_DIR), rel_path)
        gt_im = Image.open(gt_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        gt_np = np.array(gt_im).astype(np.float32) / 255.0

        try:
            psnr = compare_psnr(gt_np, rgb_pred, data_range=1.0)
            ssim = compare_ssim(gt_np, rgb_pred, data_range=1.0, channel_axis=2)
        except TypeError:
            psnr = compare_psnr(gt_np, rgb_pred, data_range=1.0)
            ssim = compare_ssim(gt_np, rgb_pred, data_range=1.0)

        rows.append((rel_path, float(psnr), float(ssim)))
        psnr_sum += psnr
        ssim_sum += ssim
        n += 1

    # save csv
    with open(out_csv, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['rel_path', 'psnr', 'ssim'])
        for r in rows:
            writer.writerow([r[0], r[1], r[2]])

    print(f'Evaluated {n} images. Avg PSNR {psnr_sum/n:.4f}, Avg SSIM {ssim_sum/n:.4f}')

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')
    parser.add_argument('--eval_ckpt', type=str, default='', help='checkpoint to evaluate on test split (optional)')
    parser.add_argument('--eval_out', type=str, default='reg_results.csv', help='csv output for eval')
    args = parser.parse_args()

    if args.eval_ckpt:
        eval_on_split(args.eval_ckpt, split_file='splits/test.txt', out_csv=args.eval_out)
    else:
        train(args)
