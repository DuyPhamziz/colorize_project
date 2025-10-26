# config.py
# config.py (Đã sửa cho Kaggle Dataset)
from pathlib import Path
import torch

# ĐỊNH NGHĨA ĐƯỜNG DẪN MỚI
KAGGLE_DATASET_NAME = 'my-colorization-data' 
DATASET_ROOT = Path('/kaggle/input') / KAGGLE_DATASET_NAME
DATA_DIR = DATASET_ROOT / 'data' # <--- Đây là đường dẫn chính xác đến thư mục chứa 'gray' và 'color'

GRAY_DIR = DATA_DIR / 'gray'
COLOR_DIR = DATA_DIR / 'color'

CHECKPOINT_DIR = Path('.').resolve() / 'checkpoints'
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
