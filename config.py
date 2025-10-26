# config.py (Đã sửa và sắp xếp lại để khắc phục NameError)
from pathlib import Path
import torch

# ************************************************
# 1. KHẮC PHỤC LỖI NAMERROR BẰNG CÁCH ĐỊNH NGHĨA ROOT Ở ĐẦU
ROOT = Path('.').resolve()
# ************************************************

# 2. ĐỊNH NGHĨA ĐƯỜNG DẪN DATASET (Đã sửa cho Kaggle Input)
KAGGLE_DATASET_NAME = 'my-colorization-data' 
DATASET_ROOT = Path('/kaggle/input') / KAGGLE_DATASET_NAME
DATA_DIR = DATASET_ROOT / 'data' # <--- Đường dẫn chính xác đến thư mục ảnh

GRAY_DIR = DATA_DIR / 'gray'
COLOR_DIR = DATA_DIR / 'color'
CHECKPOINT_DIR = ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Zhang 2016 params
NUM_BINS = 313
ANNEAL_T = 0.38

# DÒNG BỊ LỖI TRƯỚC ĐÓ, giờ sẽ hoạt động
BINS_PATH = ROOT / 'bins_313.npy'

LOG_FILE = ROOT / 'train_log.txt'