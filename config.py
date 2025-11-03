import os

# Dataset
DATA_DIR = "dataset"
IMG_SIZE = 128
NUM_WORKERS = 2

# Training
BATCH_SIZE = 4
EPOCHS = 50
LR_GEN = 2e-4
LR_DISC = 2e-4
LAMBDA_L1 = 100
LOG_INTERVAL = 50

# Directories
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
