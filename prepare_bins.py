# prepare_bins.py (Đã sửa hoàn chỉnh)
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from pathlib import Path  # Đảm bảo đã import Path
from config import COLOR_DIR, BINS_PATH

def sample_ab_pixels(color_dir, max_samples=200000, img_size=128):
    color_path = Path(color_dir) # Chuyển sang Path object
    exts = ('.jpg','.jpeg','.png')
    # SỬA Ở ĐÂY: Dùng Path.rglob thay vì os.walk
    files = [p for p in color_path.rglob('*') if p.suffix.lower() in exts]

    if not files: # Kiểm tra nếu danh sách files rỗng
        raise RuntimeError('No images found under ' + str(color_dir))

    random.shuffle(files)
    samples = []
    # Tính toán số lượng mẫu mỗi ảnh cẩn thận hơn
    num_files = len(files)
    per_image = max(5, max_samples // max(1, num_files)) if num_files > 0 else 5

    total = 0
    print(f"Sampling up to {per_image} pixels from {num_files} images...") # Thêm log

    for f_path in files: # Duyệt qua Path objects
        try:
            im = Image.open(f_path).convert('RGB')
            im = im.resize((img_size,img_size), Image.BICUBIC)
            arr = np.array(im)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            ab = lab[:,:,1:3].reshape(-1,2)

            # Lấy mẫu cẩn thận hơn
            num_pixels = ab.shape[0]
            samples_to_take = min(per_image, num_pixels) # Lấy tối đa số pixel có

            if samples_to_take < num_pixels:
                idx = np.random.choice(num_pixels, samples_to_take, replace=False)
                ab_sampled = ab[idx]
            else:
                ab_sampled = ab

            samples.append(ab_sampled)
            total += ab_sampled.shape[0]
            if total >= max_samples:
                print(f"Reached max_samples ({total}). Stopping sampling.") # Thêm log
                break
        except Exception as e:
            print('skip', str(f_path), e) # Dùng str(f_path)

    if len(samples) == 0:
        # Lỗi này không nên xảy ra nếu kiểm tra files ở trên thành công
        raise RuntimeError('Could not process any images under ' + str(color_dir))

    samples = np.vstack(samples)
    print('Collected samples shape:', samples.shape)
    return samples

def build_kmeans(n_clusters=313, out_path=BINS_PATH):
    print('Sampling ab pixels...')
    X = sample_ab_pixels(str(COLOR_DIR))
    print('Running KMeans with', n_clusters, 'clusters (this may take a while)...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    np.save(out_path, centers)
    print('Saved bins to', out_path)

if __name__ == '__main__':
    build_kmeans()