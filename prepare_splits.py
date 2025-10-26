# prepare_splits.py
"""
Usage:
  python prepare_splits.py --data_root ./data --train_pct 0.8 --val_pct 0.1 --seed 42

This script will look under <data_root>/gray and <data_root>/color and collect matching relative paths.
It emits splits/train.txt, splits/val.txt, splits/test.txt containing relative paths (relative to gray/color root),
one relative path per line (e.g. animal/00001.jpg).
"""
import argparse
from pathlib import Path
import random

def collect_rel_images(root: Path):
    exts = ('.jpg','.jpeg','.png')
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    rels = [p.relative_to(root).as_posix() for p in files]
    rels = sorted(rels)
    return rels

def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for it in items:
            f.write(it + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='root folder that contains gray/ and color/')
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--val_pct', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    gray_root = data_root / 'gray'
    color_root = data_root / 'color'

    if not gray_root.exists() or not color_root.exists():
        print('ERROR: expected directories:', gray_root, 'and', color_root)
        return

    rel_gray = collect_rel_images(gray_root)
    rel_color = collect_rel_images(color_root)

    set_gray = set(rel_gray)
    set_color = set(rel_color)
    common = sorted(list(set_gray & set_color))

    print('Found images (gray):', len(rel_gray))
    print('Found images (color):', len(rel_color))
    print('Matched pairs:', len(common))
    if len(common) == 0:
        print('No matching files between gray/ and color/. Check filenames/structure.')
        return

    random.seed(args.seed)
    random.shuffle(common)
    n = len(common)
    n_train = int(n * args.train_pct)
    n_val = int(n * args.val_pct)
    train = common[:n_train]
    val = common[n_train:n_train+n_val]
    test = common[n_train+n_val:]

    out_dir = Path('splits')
    out_dir.mkdir(exist_ok=True)
    write_list(out_dir / 'train.txt', train)
    write_list(out_dir / 'val.txt', val)
    write_list(out_dir / 'test.txt', test)

    print(f'Wrote splits: train={len(train)}, val={len(val)}, test={len(test)}')
    print('Example entries (train):', train[:3])

if __name__ == '__main__':
    main()
