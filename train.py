# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ColorizationDataset
from models.unet import UNet313
from utils import annealed_mean_from_logits, lab_to_rgb_batch, load_bins
# config may define defaults; CLI can override them
from config import DEVICE as CFG_DEVICE, BATCH_SIZE as CFG_BATCH_SIZE, LR as CFG_LR, \
    EPOCHS as CFG_EPOCHS, CHECKPOINT_DIR as CFG_CHECKPOINT_DIR, LOG_FILE as CFG_LOG_FILE
import numpy as np
import imageio
from tqdm import tqdm
import random

# speed / stability tweaks
torch.backends.cudnn.benchmark = True

def parse_args():
    p = argparse.ArgumentParser(description="Train UNet colorization (with CLI overrides for config.py defaults)")
    # general overrides (defaults fall back to config.py values)
    p.add_argument("--device", type=str, default=None, help=f"device to run on (default from config: {CFG_DEVICE})")
    p.add_argument("--batch_size", type=int, default=None, help=f"training batch size (default from config: {CFG_BATCH_SIZE})")
    p.add_argument("--val_batch_size", type=int, default=None, help="validation batch size (default: max(1, batch_size//2))")
    p.add_argument("--lr", type=float, default=None, help=f"learning rate (default from config: {CFG_LR})")
    p.add_argument("--epochs", type=int, default=None, help=f"number of epochs (default from config: {CFG_EPOCHS})")
    p.add_argument("--checkpoint_dir", type=str, default=None, help=f"folder to save ckpt (default from config: {CFG_CHECKPOINT_DIR})")
    p.add_argument("--log_file", type=str, default=None, help=f"log file path (default from config: {CFG_LOG_FILE})")
    p.add_argument("--workers", type=int, default=4, help="num_workers for dataloaders")
    p.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs")
    p.add_argument("--save_sample_every", type=int, default=1, help="save validation sample every N epochs (0=never)")
    p.add_argument("--amp", action="store_true", help="use mixed precision (torch.cuda.amp)")
    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps (>=1)")
    p.add_argument("--fast", action="store_true", help="fast mode (smaller model / less IO) -- behavior left to model impl")
    p.add_argument("--pin_memory", action="store_true", help="use pin_memory for DataLoader")
    p.add_argument("--no_pin_memory", action="store_true", help="disable pin_memory (overrides --pin_memory)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--checkpoint_prefix", type=str, default="unet_epoch", help="prefix for saved ckpt")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    p.add_argument("--workers_val", type=int, default=None, help="num_workers for val loader (default: max(1, workers//2))")
    return p.parse_args()

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    # resolve config defaults vs CLI overrides
    DEVICE = args.device if args.device is not None else CFG_DEVICE
    BATCH_SIZE = args.batch_size if args.batch_size is not None else CFG_BATCH_SIZE
    LR = args.lr if args.lr is not None else CFG_LR
    EPOCHS = args.epochs if args.epochs is not None else CFG_EPOCHS
    CHECKPOINT_DIR = args.checkpoint_dir if args.checkpoint_dir is not None else CFG_CHECKPOINT_DIR
    LOG_FILE = args.log_file if args.log_file is not None else CFG_LOG_FILE

    os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)

    # require splits created by prepare_splits.py
    train_file = 'splits/train.txt'
    val_file = 'splits/val.txt'
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise RuntimeError('Please run prepare_splits.py to generate splits/train.txt and splits/val.txt')

    # optional seed
    set_seed(args.seed)

    # dataset
    train_ds = ColorizationDataset(train_file)
    val_ds = ColorizationDataset(val_file)

    # bins
    try:
        bins = load_bins()
    except Exception as e:
        raise RuntimeError('Bins file not found. Run prepare_bins.py to create bins_313.npy before training classification.') from e

    # dataloaders
    pin_memory = True if args.pin_memory and not args.no_pin_memory else False
    val_batch = args.val_batch_size if args.val_batch_size is not None else max(1, BATCH_SIZE // 2)
    workers_val = args.workers_val if args.workers_val is not None else max(1, args.workers // 2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                            num_workers=workers_val, pin_memory=pin_memory)

    # model
    # allow 'fast' to be passed to model if the model supports it
    try:
        model = UNet313(n_bins=bins.shape[0], fast=args.fast).to(DEVICE)
    except TypeError:
        # fallback if model doesn't accept fast param
        model = UNet313(n_bins=bins.shape[0]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # simple scheduler
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and DEVICE != 'cpu')

    # resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=DEVICE)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)
            if 'optimizer' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except Exception:
                    print("⚠️ Could not load optimizer state (optimizer mismatch). Continuing without it.")
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"🔁 Resumed training from {args.resume} (starting epoch {start_epoch})")
        else:
            print(f"⚠️ Resume path {args.resume} not found. Starting from scratch.")

    # print config summary
    print("========== Training configuration ==========")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}  | Start epoch: {start_epoch}")
    print(f"Batch size: {BATCH_SIZE}  | Val batch: {val_batch}")
    print(f"LR: {LR}  | Accum steps: {args.accum_steps}  | AMP: {args.amp}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}  | Prefix: {args.checkpoint_prefix}")
    print(f"Save every {args.save_every} epochs  | Save sample every {args.save_sample_every} epochs")
    print(f"Workers: train={args.workers} val={workers_val}  | pin_memory={pin_memory}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("============================================")

    global_step = 0
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)
        optimizer.zero_grad()
        for i, (L, target, _) in pbar:
            L = L.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            device_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
            with torch.amp.autocast(device_type, enabled=args.amp and DEVICE != 'cpu'):

                logits = model(L)  # B x C x H x W
                B, C, H, W = logits.shape
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
                targ_flat = target.view(-1)
                loss = criterion(logits_flat, targ_flat) / args.accum_steps

            scaler.scale(loss).backward()

            # gradient accumulation step
            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            total_loss += (loss.item() * args.accum_steps)
            batch_count += 1
            pbar.set_postfix(loss=f"{(total_loss / batch_count):.4f}")

        avg_loss = total_loss / max(1, batch_count)
        print(f"✅ Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.6f}")

        # Validation preview (save sample if requested)
        model.eval()
        with torch.no_grad():
            try:
                Lval, _, rels = next(iter(val_loader))
                Lval = Lval.to(DEVICE)
                logits_v = model(Lval)
                ab_v = annealed_mean_from_logits(logits_v, bins)
                rgb = lab_to_rgb_batch(Lval, ab_v)
                if args.save_sample_every and args.save_sample_every > 0 and (epoch % args.save_sample_every == 0):
                    sample_out = os.path.join(str(CHECKPOINT_DIR), f'epoch{epoch}_sample.png')
                    # save first image only (assume rgb is HxWx3 or batch)
                    # if rgb is batch, index 0
                    img_to_save = rgb[0] if isinstance(rgb, (list, tuple, np.ndarray)) and getattr(rgb, 'ndim', None) != 3 else rgb
                    imageio.imwrite(sample_out, img_to_save)
                    print(f"📸 Saved validation sample: {sample_out}")
            except StopIteration:
                print("⚠️ Validation loader empty.")
            except Exception as e:
                print(f"⚠️ Error during validation preview: {e}")

        # scheduler step
        try:
            scheduler.step()
        except Exception:
            pass

        # Save checkpoint every save_every epochs
        if args.save_every and (epoch % args.save_every == 0):
            ck = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            out_path = os.path.join(str(CHECKPOINT_DIR), f'{args.checkpoint_prefix}{epoch}.pth')
            torch.save(ck, out_path)
            print(f"💾 Saved checkpoint: {out_path}")

        # append log
        try:
            with open(str(LOG_FILE), 'a', encoding='utf8') as f:
                f.write(f'{epoch}\t{avg_loss:.6f}\n')
        except Exception:
            pass

    print("🎉 Training completed successfully!")

if __name__ == '__main__':
    args = parse_args()
    train(args)
