# infer.py
import sys
import os
import torch
import imageio
import numpy as np
from PIL import Image
from datasets import ColorizationDataset
from models.unet import UNet313
from utils import load_bins, annealed_mean_from_logits, lab_to_rgb_batch
from config import DEVICE, COLOR_DIR, IMG_SIZE

# optional metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

def _load_ckpt_model(checkpoint):
    bins = load_bins()
    model = UNet313(n_bins=bins.shape[0]).to(DEVICE)
    ck = torch.load(checkpoint, map_location=DEVICE)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.eval()
    return model, bins

def _load_gt(rel_path):
    """
    Trả về numpy array float32 trong [0,1] kích thước (H,W,3).
    Nếu file GT không tìm thấy -> trả về None.
    """
    if not COLOR_DIR:
        return None
    gt_path = os.path.join(str(COLOR_DIR), rel_path)
    if not os.path.exists(gt_path):
        return None
    im = Image.open(gt_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.array(im).astype(np.float32) / 255.0
    return arr

def _compute_metrics(gt_np, pred_np):
    """
    gt_np, pred_np: numpy float32 in [0,1], shape (H,W,3)
    Trả về psnr, ssim (float). Nếu skimage không có -> trả None, None
    """
    if not HAVE_SKIMAGE:
        return None, None
    try:
        p = compare_psnr(gt_np, pred_np, data_range=1.0)
        s = compare_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)
    except TypeError:
        # older skimage may not accept channel_axis
        p = compare_psnr(gt_np, pred_np, data_range=1.0)
        s = compare_ssim(gt_np, pred_np, data_range=1.0)
    return float(p), float(s)

def infer_one(checkpoint, idx=0, out='infer_out.png', write_metrics=False, metrics_file='infer_metrics.txt'):
    model, bins = _load_ckpt_model(checkpoint)

    ds = ColorizationDataset('splits/test.txt')
    L, _, rel = ds[idx]
    L = L.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(L)
        ab = annealed_mean_from_logits(logits, bins)
        rgb = lab_to_rgb_batch(L, ab)[0]  # likely uint8 HxWx3 [0..255]

    # save image
    imageio.imwrite(out, rgb)
    print('Saved', out)

    # optional metrics
    if write_metrics:
        rel_path = rel if isinstance(rel, str) else rel[0]
        gt = _load_gt(rel_path)
        if gt is None:
            print('GT not found or COLOR_DIR not set -> skipping metrics')
            return
        pred = rgb.astype(np.float32) / 255.0
        psnr, ssim = _compute_metrics(gt, pred)
        if psnr is None:
            print('skimage not available -> cannot compute PSNR/SSIM')
            return
        # append to metrics file
        with open(metrics_file, 'a', encoding='utf8') as f:
            f.write(f"{rel_path}\t{psnr:.4f}\t{ssim:.4f}\n")
        print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f} -> appended to {metrics_file}")

def infer_all(checkpoint, out_dir='infer_results', metrics_csv='results_infer.csv'):
    """
    Infer toàn bộ dataset splits/test.txt.
    - Lưu ảnh ra out_dir (tên file là basename(rel).png)
    - Nếu tìm được GTs trong COLOR_DIR và skimage có sẵn -> tính PSNR/SSIM và ghi CSV
    """
    os.makedirs(out_dir, exist_ok=True)
    model, bins = _load_ckpt_model(checkpoint)
    ds = ColorizationDataset('splits/test.txt')

    rows = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    compute_metrics_flag = HAVE_SKIMAGE and COLOR_DIR is not None and str(COLOR_DIR) != ''

    with torch.no_grad():
        for i, (L, _, rel) in enumerate(ds):
            L = L.unsqueeze(0).to(DEVICE)
            logits = model(L)
            ab = annealed_mean_from_logits(logits, bins)
            rgb = lab_to_rgb_batch(L, ab)[0]  # HxWx3 uint8
            # prepare filenames
            rel_path = rel if isinstance(rel, str) else rel[0]
            name = os.path.splitext(os.path.basename(rel_path))[0]
            out_path = os.path.join(out_dir, f'{name}.png')
            imageio.imwrite(out_path, rgb)
            print(f'Saved {out_path}')

            # try compute metrics
            if compute_metrics_flag:
                gt = _load_gt(rel_path)
                if gt is not None:
                    pred = rgb.astype(np.float32) / 255.0
                    psnr, ssim = _compute_metrics(gt, pred)
                    if psnr is not None:
                        rows.append((rel_path, psnr, ssim))
                        psnr_sum += psnr
                        ssim_sum += ssim
                        n += 1

    # write CSV if any metrics computed
    if rows:
        import csv
        with open(metrics_csv, 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['rel_path','psnr','ssim'])
            for r in rows:
                writer.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])
            writer.writerow(['Average', f"{(psnr_sum/n):.4f}", f"{(ssim_sum/n):.4f}"])
        print(f'Evaluated {n} images. Avg PSNR: {psnr_sum/n:.4f}, Avg SSIM: {ssim_sum/n:.4f}')
        print(f'Results written to {metrics_csv}')
    else:
        if compute_metrics_flag:
            print('No GTs matched in COLOR_DIR -> no metrics written.')
        else:
            if not HAVE_SKIMAGE:
                print('skimage not installed -> cannot compute PSNR/SSIM.')
            if not COLOR_DIR:
                print('COLOR_DIR not set in config -> cannot find GTs.')

if __name__ == '__main__':
    """
    Usage:
      python infer.py <checkpoint> [index|all] [out.png]
    Behavior:
      - "all" : infer toàn bộ dataset -> lưu vào ./infer_results. Nếu tìm được GTs và skimage có sẵn thì cũng tính PSNR/SSIM và ghi results_infer.csv
      - <index> : infer 1 ảnh ở chỉ số index, lưu ra out.png
      - nếu muốn khi infer 1 ảnh đồng thời tính metrics, thêm "eval" vào argument cuối, ví dụ:
          python infer.py checkpoints/model.pth 0 out.png eval
    """
    if len(sys.argv) < 2:
        print('Usage: python infer.py <checkpoint> [index|all] [out.png] [eval]')
        sys.exit(1)

    ck = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else '0'

    # if user asked 'all'
    if mode == 'all':
        infer_all(ck)
    else:
        # mode treated as index
        try:
            idx = int(mode)
        except ValueError:
            idx = 0
        out = sys.argv[3] if len(sys.argv) > 3 else 'infer_out.png'
        eval_flag = (len(sys.argv) > 4 and sys.argv[4].lower() in ('1', 'true', 'eval'))
        infer_one(ck, idx=idx, out=out, write_metrics=eval_flag)
