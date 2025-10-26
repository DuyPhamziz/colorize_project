# colorize_project

Steps:
1. Place dataset under `data/gray/...` and `data/color/...` with matching relative paths.
2. Create splits:
   `python prepare_splits.py --gray_root data/gray --train_pct 0.8 --val_pct 0.1`
3. Build 313 bins:
   `python prepare_bins.py`
4. Train:
   `python train.py`
5. Evaluate on test:
   `python eval_testset.py checkpoints/unet_epoch60.pth results_test.csv`
6. Infer single:
   `python infer.py checkpoints/unet_epoch60.pth 0 out.png`
