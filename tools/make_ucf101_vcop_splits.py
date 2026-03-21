import json
import random
from pathlib import Path
from collections import defaultdict

seed = 632
random.seed(seed)

json_path = Path("/projects/u5ia/pxl416/data/UCF-101/ucf101_non_sign_index.json")
out_root = Path("/projects/u5ia/pxl416/data/ucf101_vcop")
split_dir = out_root / "split"
split_dir.mkdir(parents=True, exist_ok=True)

clip_len = 16
interval = 8
tuple_len = 3
min_frames = clip_len * tuple_len + interval * (tuple_len - 1)  # 64

with open(json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

videos = meta["videos"]
class_names = sorted(meta["classes"].keys())

# 1) classInd.txt
with open(split_dir / "classInd.txt", "w", encoding="utf-8") as f:
    for i, cls in enumerate(class_names, start=1):
        f.write(f"{i} {cls}\n")

# 2) group videos by class
by_class = defaultdict(list)
for item in videos:
    by_class[item["class"]].append(item)

for cls in by_class:
    by_class[cls] = sorted(by_class[cls], key=lambda x: x["relpath"])
    random.shuffle(by_class[cls])

train_relpaths = []
test_relpaths = []

# 3) stratified split per class (80/20)
for cls in class_names:
    items = by_class[cls]
    n = len(items)
    n_train = max(1, int(0.8 * n))
    train_items = items[:n_train]
    test_items = items[n_train:] if n_train < n else items[-1:]

    train_relpaths.extend([x["relpath"] for x in train_items])
    test_relpaths.extend([x["relpath"] for x in test_items])

train_relpaths = sorted(train_relpaths)
test_relpaths = sorted(test_relpaths)

# 4) save trainlist01.txt / testlist01.txt
with open(split_dir / "trainlist01.txt", "w", encoding="utf-8") as f:
    for p in train_relpaths:
        f.write(p + "\n")

with open(split_dir / "testlist01.txt", "w", encoding="utf-8") as f:
    for p in test_relpaths:
        f.write(p + "\n")

# 5) build a relpath -> num_frames map
num_frames_map = {item["relpath"]: item["num_frames"] for item in videos}

# 6) filter for VCOP minimum length
vcop_train = [p for p in train_relpaths if num_frames_map.get(p, 0) >= min_frames]
vcop_test = [p for p in test_relpaths if num_frames_map.get(p, 0) >= min_frames]

with open(split_dir / f"vcop_train_{clip_len}_{interval}_{tuple_len}.txt", "w", encoding="utf-8") as f:
    for p in vcop_train:
        f.write(p + "\n")

with open(split_dir / f"vcop_test_{clip_len}_{interval}_{tuple_len}.txt", "w", encoding="utf-8") as f:
    for p in vcop_test:
        f.write(p + "\n")

print("Done.")
print(f"class count: {len(class_names)}")
print(f"train videos: {len(train_relpaths)}")
print(f"test videos: {len(test_relpaths)}")
print(f"vcop train videos (num_frames>={min_frames}): {len(vcop_train)}")
print(f"vcop test videos (num_frames>={min_frames}): {len(vcop_test)}")