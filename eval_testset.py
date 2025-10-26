# eval_testset.py
import os
import torch
from torch.utils.data import DataLoader
from datasets import ColorizationDataset
from models.unet import UNet313
from utils import load_bins, annealed_mean_from_logits, lab_to_rgb_batch
from config import DEVICE, BATCH_SIZE, COLOR_DIR, IMG_SIZE
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import csv
from tqdm import tqdm
from PIL import Image

def evaluate(checkpoint, out_csv='results_test.csv'):
    bins = load_bins()
    model = UNet313(n_bins=bins.shape[0]).to(DEVICE)
    ck = torch.load(checkpoint, map_location=DEVICE)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.eval()

    ds = ColorizationDataset('splits/test.txt')
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    rows = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0

    for L, target, rel in tqdm(loader, desc='Eval test'):
        L = L.to(DEVICE)
        with torch.no_grad():
            logits = model(L)
            ab_pred = annealed_mean_from_logits(logits, bins)
            rgb_pred = lab_to_rgb_batch(L, ab_pred)[0].astype(np.float32) / 255.0

        # load GT resized
        rel_path = rel if isinstance(rel, str) else rel[0]
        gt_path = os.path.join(str(COLOR_DIR), rel_path)
        gt_im = Image.open(gt_path).convert('RGB').resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)
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

    with open(out_csv, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['rel_path','psnr','ssim'])
        for r in rows:
            writer.writerow([r[0], r[1], r[2]])

    print(f'Evaluated {n} images. Avg PSNR: {psnr_sum/n:.4f}, Avg SSIM: {ssim_sum/n:.4f}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python eval_testset.py <checkpoint.pth> [out.csv]')
    else:
        ck = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else 'results_test.csv'
        evaluate(ck, out)
