# prepare_bins.py
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from pathlib import Path
from config import COLOR_DIR, BINS_PATH

def sample_ab_pixels(color_dir, max_samples=200000, img_size=128):
    files = []
    for root, dirs, fnames in os.walk(color_dir):
        for f in fnames:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                files.append(os.path.join(root,f))
    random.shuffle(files)
    samples = []
    per_image = max(5, max_samples // max(1, len(files)))
    total = 0
    for f in files:
        try:
            im = Image.open(f).convert('RGB')
            im = im.resize((img_size,img_size), Image.BICUBIC)
            arr = np.array(im)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            ab = lab[:,:,1:3].reshape(-1,2)
            if per_image < ab.shape[0]:
                idx = np.random.choice(ab.shape[0], per_image, replace=False)
                ab = ab[idx]
            samples.append(ab)
            total += ab.shape[0]
            if total >= max_samples:
                break
        except Exception as e:
            print('skip', f, e)
    if len(samples) == 0:
        raise RuntimeError('No images found under ' + str(color_dir))
    samples = np.vstack(samples)
    print('Collected samples', samples.shape)
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
