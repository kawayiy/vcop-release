"""
Example usage:

    python tools/make_full_phoenix_vcop_splits.py

    python tools/make_full_phoenix_vcop_splits.py ^
      --root_dir "D:/python_projects/Uni-SLM-data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
"""

import argparse
import csv
from pathlib import Path

DEFAULT_ROOT_DIR = Path(r"D:/python_projects/Uni-SLM-data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")
DEFAULT_FEATURES_DIR = "features/fullFrame-210x260px"
DEFAULT_ANNOTATIONS_DIR = "annotations/manual"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PHOENIX VCOP split files")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="directory containing PHOENIX-2014-T annotations/ and features/",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=DEFAULT_FEATURES_DIR,
        help="relative or absolute frame directory containing train/dev/test folders",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="relative or absolute annotation directory containing PHOENIX corpus CSV files",
    )
    parser.add_argument("--clip_len", type=int, default=16, help="clip length")
    parser.add_argument("--interval", type=int, default=8, help="frames between clips")
    parser.add_argument("--tuple_len", type=int, default=3, help="number of clips in one tuple")
    parser.add_argument("--strict", action="store_true", help="raise on missing folders or unreadable samples")
    return parser.parse_args()


def resolve_dataset_root(root_dir):
    direct_features = root_dir / DEFAULT_FEATURES_DIR
    direct_annotations = root_dir / DEFAULT_ANNOTATIONS_DIR
    if direct_features.exists() and direct_annotations.exists():
        return root_dir

    nested_root = root_dir / "PHOENIX-2014-T"
    nested_features = nested_root / DEFAULT_FEATURES_DIR
    nested_annotations = nested_root / DEFAULT_ANNOTATIONS_DIR
    if nested_features.exists() and nested_annotations.exists():
        return nested_root

    return root_dir


def resolve_path(dataset_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return dataset_root / path


def load_annotation_rows(annotations_root, split_name):
    csv_path = annotations_root / f"PHOENIX-2014-T.{split_name}.corpus.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    normalized_rows = []
    for row in rows:
        sample_name = (row.get("name") or "").strip()
        if not sample_name:
            raise ValueError(f"Missing PHOENIX sample name in {csv_path}")
        row = dict(row)
        row["name"] = sample_name
        row["split"] = split_name
        normalized_rows.append(row)
    return normalized_rows


def count_frames(sample_dir):
    png_frames = list(sample_dir.glob("*.png"))
    if png_frames:
        return len(png_frames)

    jpg_frames = list(sample_dir.glob("*.jpg"))
    if jpg_frames:
        return len(jpg_frames)

    jpeg_frames = list(sample_dir.glob("*.jpeg"))
    return len(jpeg_frames)


def output_split_name(source_split_name):
    if source_split_name == "dev":
        return "val"
    return source_split_name


def write_split_file(split_dir, split_name, sample_names, clip_len, interval, tuple_len):
    split_path = split_dir / f"vcop_{split_name}_{clip_len}_{interval}_{tuple_len}.txt"
    with open(split_path, "w", encoding="utf-8") as f:
        for sample_name in sample_names:
            f.write(sample_name + "\n")
    return split_path


def filter_split(rows, features_root, min_frames, strict):
    kept = []
    skipped_missing = []
    skipped_short = []
    skipped_empty = []

    for row in rows:
        sample_dir = features_root / row["split"] / row["name"]
        if not sample_dir.exists():
            if strict:
                raise FileNotFoundError(f"PHOENIX frame folder not found: {sample_dir}")
            skipped_missing.append(row["name"])
            continue

        frame_count = count_frames(sample_dir)
        if frame_count <= 0:
            if strict:
                raise ValueError(f"No frame images found in: {sample_dir}")
            skipped_empty.append(row["name"])
            continue

        if frame_count < min_frames:
            skipped_short.append(row["name"])
            continue

        kept.append(row["name"])

    return kept, skipped_missing, skipped_short, skipped_empty


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    dataset_root = resolve_dataset_root(root_dir)
    features_root = resolve_path(dataset_root, args.features_dir)
    annotations_root = resolve_path(dataset_root, args.annotations_dir)
    split_dir = dataset_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    min_frames = args.clip_len * args.tuple_len + args.interval * (args.tuple_len - 1)
    stats = {}

    for source_split_name in ("train", "dev", "test"):
        rows = load_annotation_rows(annotations_root, source_split_name)
        kept, skipped_missing, skipped_short, skipped_empty = filter_split(
            rows,
            features_root,
            min_frames,
            args.strict,
        )
        split_name = output_split_name(source_split_name)
        split_path = write_split_file(
            split_dir,
            split_name,
            kept,
            args.clip_len,
            args.interval,
            args.tuple_len,
        )
        stats[split_name] = {
            "annotation_rows": len(rows),
            "kept": len(kept),
            "missing": len(skipped_missing),
            "short": len(skipped_short),
            "empty": len(skipped_empty),
            "split_path": split_path,
        }

    print("Done.")
    print("dataset_root:", dataset_root)
    print("annotations_root:", annotations_root)
    print("features_root:", features_root)
    print("min_frames:", min_frames)
    for split_name in ("train", "val", "test"):
        split_stats = stats[split_name]
        print(
            "{}: annotation_rows={}, kept={}, missing={}, short={}, empty={}, split={}".format(
                split_name,
                split_stats["annotation_rows"],
                split_stats["kept"],
                split_stats["missing"],
                split_stats["short"],
                split_stats["empty"],
                split_stats["split_path"],
            )
        )


if __name__ == "__main__":
    main()
