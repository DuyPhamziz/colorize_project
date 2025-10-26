# config.py
from pathlib import Path
import torch

# Thay đổi: Định nghĩa đường dẫn tuyệt đối đến Kaggle Dataset
KAGGLE_DATASET_NAME = 'my-colorization-data' # Đổi tên này nếu tên dataset của bạn khác!
DATASET_PATH = Path('/kaggle/input') / KAGGLE_DATASET_NAME
# DATASET_PATH sẽ là: /kaggle/input/my-colorization-data/

ROOT = Path('.').resolve() # Giữ lại để định nghĩa nơi lưu checkpoint/log
DATA_DIR = DATASET_PATH
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

BINS_PATH = ROOT / 'bins_313.npy'

LOG_FILE = ROOT / 'train_log.txt'
