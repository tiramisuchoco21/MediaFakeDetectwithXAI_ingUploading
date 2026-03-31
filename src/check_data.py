import os

train_real_path = "data/train/real"
train_fake_path = "data/train/fake"
val_real_path = "data/val/real"
val_fake_path = "data/val/fake"

train_real_files = os.listdir(train_real_path) if os.path.exists(train_real_path) else []
train_fake_files = os.listdir(train_fake_path) if os.path.exists(train_fake_path) else []
val_real_files = os.listdir(val_real_path) if os.path.exists(val_real_path) else []
val_fake_files = os.listdir(val_fake_path) if os.path.exists(val_fake_path) else []

print(f"Train Real 이미지 개수: {len(train_real_files)}장")
print(f"Train Fake 이미지 개수: {len(train_fake_files)}장")
print(f"Val   Real 이미지 개수: {len(val_real_files)}장")
print(f"Val   Fake 이미지 개수: {len(val_fake_files)}장")
