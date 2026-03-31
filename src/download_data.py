from datasets import load_dataset
import os

os.makedirs('data/train/real', exist_ok=True)
os.makedirs('data/train/fake', exist_ok=True)
os.makedirs('data/val/real', exist_ok=True)
os.makedirs('data/val/fake', exist_ok=True)

ds = load_dataset(
    'Parveshiiii/AI-vs-Real',
    split='train',
    streaming=True
)

real_count, fake_count = 0, 0
MAX = 3000

for sample in ds:
    label = sample['binary_label']
    img   = sample['image']

    if label == 1 and real_count < MAX:
        img.save(f'data/train/real/{real_count}.jpg')
        real_count += 1
    elif label == 0 and fake_count < MAX:
        img.save(f'data/train/fake/{fake_count}.jpg')
        fake_count += 1

    if real_count >= MAX and fake_count >= MAX/2:
        break

print(f'최종 real: {real_count}  fake: {fake_count}')
print('완료!')


real_count, fake_count = 0, 0
MAX = 400 # real 400, fake 400 = 총 800개 (train의 20%)
SKIP = MAX # 앞에서 train으로 쓴 MAX개 건너뛰기

for i, sample in enumerate(ds):
    if i < SKIP * 2: # 앞 2*MAX개 skip
        continue

    label = sample['binary_label']
    img   = sample['image']

    if label == 1 and real_count < MAX:
        img.save(f'data/val/real/{real_count}.jpg')
        real_count += 1
    elif label == 0 and fake_count < MAX:
        img.save(f'data/val/fake/{fake_count}.jpg')
        fake_count += 1

    if real_count >= MAX and fake_count >= MAX:
        break

print(f'val real: {real_count}  fake: {fake_count}')
print('완료!')
