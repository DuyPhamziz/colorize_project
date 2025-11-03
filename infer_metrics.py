# infer_metrics.py
import os
import csv
import numpy as np

# Thư mục lưu infer_log.txt
OUTPUT_DIR = "outputs"
log_file = os.path.join(OUTPUT_DIR, "infer_log.txt")
summary_file = os.path.join(OUTPUT_DIR, "infer_summary.txt")

if not os.path.exists(log_file):
    print(f"File {log_file} không tồn tại!")
    exit(1)

psnr_list = []
ssim_list = []
mse_list = []

# Đọc dữ liệu từ infer_log.txt
with open(log_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            psnr_list.append(float(row['PSNR']))
            ssim_list.append(float(row['SSIM']))
            mse_list.append(float(row['MSE']))
        except ValueError:
            continue  # Bỏ qua dòng không hợp lệ

# Tính trung bình
psnr_mean = np.mean(psnr_list) if psnr_list else 0
ssim_mean = np.mean(ssim_list) if ssim_list else 0
mse_mean = np.mean(mse_list) if mse_list else 0

# Ghi ra file summary
with open(summary_file, "w") as f:
    f.write("Metric,Mean\n")
    f.write(f"PSNR,{psnr_mean:.4f}\n")
    f.write(f"SSIM,{ssim_mean:.4f}\n")
    f.write(f"MSE,{mse_mean:.6f}\n")

print(f"Tổng hợp xong! Trung bình ghi vào {summary_file}")
print(f"PSNR={psnr_mean:.4f}, SSIM={ssim_mean:.4f}, MSE={mse_mean:.6f}")
