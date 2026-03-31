import kagglehub

def download_latest_ver() :
    path = kagglehub.dataset_download("khafie0212/dfdc-dataset")
    return f"Path to dataset files: {path}"

download_latest_ver()

import os
import shutil
import pandas as pd

train_limits = {"real": 1000, "fake": 500}
val_limits = {"real": 100, "fake": 100}

DFDC_BASE = "/home/work/.cache/kagglehub/datasets/khafie0212/dfdc-dataset/versions/1/dfdc_frames"
train_DATA_DIR  = "data/train"
val_DATA_DIR  = "data/val"

os.makedirs(f"{train_DATA_DIR}/real", exist_ok=True)
os.makedirs(f"{train_DATA_DIR}/fake", exist_ok=True)
os.makedirs(f"{val_DATA_DIR}/real", exist_ok=True)
os.makedirs(f"{val_DATA_DIR}/fake", exist_ok=True)

# train
for label in ["real", "fake"]:
    src_dir = f"{DFDC_BASE}/train/{label}"
    dst_dir = f"{train_DATA_DIR}/{label}"

    if not os.path.exists(src_dir):
        print(f"{src_dir} 없음")
        continue

    train_limit = train_limits[label]
    files = os.listdir(src_dir)[:train_limit]

    for fname in files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, f"dfdc_{fname}")
        shutil.copy2(src, dst)

    print(f"Train {label}: {len(files)}장 복사 완료")

# val
for label in ["real", "fake"]:
    src_dir = f"{DFDC_BASE}/val/{label}"
    dst_dir = f"{val_DATA_DIR}/{label}"

    if not os.path.exists(src_dir):
        print(f"{src_dir} 없음")
        continue

    val_limit = val_limits[label]
    files = os.listdir(src_dir)[:val_limit]

    for fname in files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, f"dfdc_{fname}")
        shutil.copy2(src, dst)

    print(f"Val {label}: {len(files)}장 복사 완료")
