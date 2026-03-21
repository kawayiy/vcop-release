import os

input_file = "/projects/u5ia/pxl416/data/CSL-Daily/sentence_label/split_1.txt"
output_file = "/projects/u5ia/pxl416/data/CSL-Daily/sentence_label/all_ids_clean.txt"

seen = set()
ordered_ids = []

with open(input_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        # 跳过表头
        if line.lower() == "name|split":
            continue

        if "|" not in line:
            print(f"[WARN] bad format at line {line_no}: {line}")
            continue

        sample_id, split = line.split("|", 1)
        sample_id = sample_id.strip()

        if not sample_id:
            continue

        if sample_id in seen:
            continue

        seen.add(sample_id)
        ordered_ids.append(sample_id)

with open(output_file, "w", encoding="utf-8") as f:
    for sid in ordered_ids:
        f.write(sid + "\n")

print(f"Saved clean all_ids: {len(ordered_ids)} samples")
print(f"Path: {output_file}")