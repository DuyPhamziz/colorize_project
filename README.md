# colorize_project

Steps:
#!/bin/bash
echo "=== B1: Chuẩn bị splits ==="
python prepare_splits.py --gray_root data/gray --train_pct 0.8 --val_pct 0.1

echo "=== B2: Chuẩn bị bins ==="
python prepare_bins.py

echo "=== B3: Huấn luyện model (AMP mode) ==="
python train.py --batch_size 16 --lr 1e-4 --epochs 60

echo "=== B4: Inference toàn bộ test set ==="
python infer.py checkpoints/unet_epoch30.pth all

echo "=== B5: Đánh giá kết quả ==="
python eval_testset.py checkpoints/unet_epoch30.pth results_test.csv