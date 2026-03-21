from pathlib import Path
import random

seed = 632
random.seed(seed)

root = Path("data/mini_CSL_Daily")
sent_dir = root / "sentence"
split_dir = root / "split"
split_dir.mkdir(parents=True, exist_ok=True)

clip_len = 16
interval = 8
tuple_len = 3
min_frames = clip_len * tuple_len + interval * (tuple_len - 1)  # 64

sample_dirs = sorted([p for p in sent_dir.iterdir() if p.is_dir()])

def count_frames(sample_dir: Path) -> int:
    return len(list(sample_dir.glob("*.jpg")))

all_names = [p.name for p in sample_dirs]
random.shuffle(all_names)

n = len(all_names)
n_train = max(1, int(0.8 * n))
train_names = sorted(all_names[:n_train])
test_names = sorted(all_names[n_train:])

with open(split_dir / "trainlist01.txt", "w", encoding="utf-8") as f:
    for name in train_names:
        f.write(name + "\n")

with open(split_dir / "testlist01.txt", "w", encoding="utf-8") as f:
    for name in test_names:
        f.write(name + "\n")

name_to_frames = {p.name: count_frames(p) for p in sample_dirs}

vcop_train = [name for name in train_names if name_to_frames[name] >= min_frames]
vcop_test = [name for name in test_names if name_to_frames[name] >= min_frames]

with open(split_dir / f"vcop_train_{clip_len}_{interval}_{tuple_len}.txt", "w", encoding="utf-8") as f:
    for name in vcop_train:
        f.write(name + "\n")

with open(split_dir / f"vcop_test_{clip_len}_{interval}_{tuple_len}.txt", "w", encoding="utf-8") as f:
    for name in vcop_test:
        f.write(name + "\n")

print("Done.")
print("total samples:", len(all_names))
print("train:", len(train_names))
print("test:", len(test_names))
print(f"vcop train (frames>={min_frames}):", len(vcop_train))
print(f"vcop test (frames>={min_frames}):", len(vcop_test))