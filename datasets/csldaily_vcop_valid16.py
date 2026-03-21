from pathlib import Path

data_root = Path("/projects/u5ia/pxl416/data/CSL-Daily")
sentence_dir = data_root / "sentence"
input_txt = data_root / "sentence_label" / "all_ids_clean.txt"
output_txt = data_root / "sentence_label" / "all_ids_vcop_valid_cl16.txt"
report_txt = data_root / "sentence_label" / "all_ids_vcop_valid_cl16_report.txt"

clip_len = 16
interval = 8
tuple_len = 3
min_frames = clip_len * tuple_len + interval * (tuple_len - 1)  # 64

valid_ids = []
too_short = []
missing = []

with open(input_txt, "r", encoding="utf-8") as f:
    sample_ids = [line.strip() for line in f if line.strip()]

for sid in sample_ids:
    sample_dir = sentence_dir / sid
    if not sample_dir.exists():
        missing.append(sid)
        continue

    frame_paths = sorted(sample_dir.glob("*.jpg"))
    n_frames = len(frame_paths)

    if n_frames < min_frames:
        too_short.append((sid, n_frames))
    else:
        valid_ids.append(sid)

with open(output_txt, "w", encoding="utf-8") as f:
    for sid in valid_ids:
        f.write(sid + "\n")

with open(report_txt, "w", encoding="utf-8") as f:
    f.write(f"min_frames_required: {min_frames}\n")
    f.write(f"input_total: {len(sample_ids)}\n")
    f.write(f"valid_total: {len(valid_ids)}\n")
    f.write(f"missing_total: {len(missing)}\n")
    f.write(f"too_short_total: {len(too_short)}\n\n")

    f.write("[missing]\n")
    for sid in missing:
        f.write(f"{sid}\n")

    f.write("\n[too_short]\n")
    for sid, n in too_short:
        f.write(f"{sid}\t{n}\n")

print(f"min_frames_required = {min_frames}")
print(f"input_total  = {len(sample_ids)}")
print(f"valid_total  = {len(valid_ids)}")
print(f"missing_total = {len(missing)}")
print(f"too_short_total = {len(too_short)}")
print(f"saved valid ids to: {output_txt}")
print(f"saved report to: {report_txt}")