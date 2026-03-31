"""
Example usage:

    python tools/make_full_bobsl_vcop_splits.py

    python tools/make_full_bobsl_vcop_splits.py ^
      --root_dir "D:/python_projects/Uni-SLM-data/mini_BOBSL" ^
      --videos_dir "D:/python_projects/Uni-SLM-data/mini_BOBSL/original_data/videos/mp4"
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

DEFAULT_ROOT_DIR = Path(r"D:/python_projects/Uni-SLM-data/mini_BOBSL")
DEFAULT_VIDEOS_DIR = "original_data/videos/mp4"
DEFAULT_ANNOTATIONS_DIR = "manual_annotations/continuous_sign_sequences/cslr-json-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BOBSL VCOP split files")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="directory containing manifest.json and BOBSL annotations",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default=DEFAULT_VIDEOS_DIR,
        help="relative or absolute video directory",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="relative or absolute annotation directory containing VIA JSON files",
    )
    parser.add_argument("--clip_len", type=int, default=16, help="clip length")
    parser.add_argument("--interval", type=int, default=8, help="frames between clips")
    parser.add_argument("--tuple_len", type=int, default=3, help="number of clips in one tuple")
    parser.add_argument("--strict", action="store_true", help="raise on missing or unreadable videos")
    return parser.parse_args()


def resolve_videos_root(root_dir, videos_dir):
    videos_path = Path(videos_dir)
    if videos_path.is_absolute():
        return videos_path
    return root_dir / videos_path


def resolve_annotations_root(root_dir, annotations_dir):
    annotations_path = Path(annotations_dir)
    if annotations_path.is_absolute():
        return annotations_path
    return root_dir / annotations_path


def normalize_sample_name(sample_name):
    normalized_name = Path(sample_name).name.strip()
    if not normalized_name:
        raise ValueError("Encountered an empty BOBSL sample name")
    if not normalized_name.endswith(".json"):
        normalized_name += ".json"
    return normalized_name


def annotation_path(annotations_root, sample_name):
    return annotations_root / normalize_sample_name(sample_name)


def parse_time_range(src):
    if not src or "#t=" not in src:
        raise ValueError(f"Expected '#t=start,end' in BOBSL src field, but got: {src}")

    time_range = src.rsplit("#t=", 1)[1]
    start_str, end_str = time_range.split(",", 1)
    start_sec = float(start_str)
    end_sec = float(end_str)
    if end_sec <= start_sec:
        raise ValueError(f"Invalid BOBSL src time range: {src}")
    return start_sec, end_sec


def load_manifest(root_dir):
    manifest_path = root_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation(annotations_root, sample_name):
    sample_path = annotation_path(annotations_root, sample_name)
    with open(sample_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    file_entries = payload.get("file", {})
    if not file_entries:
        raise ValueError(f"No file entries found in BOBSL annotation: {sample_path}")

    first_file_entry = next(iter(file_entries.values()))
    video_name = first_file_entry.get("fname")
    if not video_name:
        raise ValueError(f"Missing fname in BOBSL annotation: {sample_path}")

    src = first_file_entry.get("src")
    start_sec, end_sec = parse_time_range(src)
    return {
        "sample_name": normalize_sample_name(sample_name),
        "sample_path": sample_path,
        "video_name": Path(video_name).name,
        "clip_time_range": (start_sec, end_sec),
    }


def resolve_video_path(root_dir, videos_root, video_name):
    primary_path = videos_root / video_name
    if primary_path.exists():
        return primary_path

    fallback_path = root_dir / video_name
    if fallback_path.exists():
        return fallback_path

    return primary_path


def probe_video_info(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0:
        raise ValueError(f"Could not infer FPS from OpenCV metadata for {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Could not infer frame count from OpenCV metadata for {video_path}")

    return fps, total_frames


def available_window_frames(video_path, start_sec, end_sec):
    fps, total_frames = probe_video_info(video_path)
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = min(total_frames, int(math.ceil(end_sec * fps)))
    return end_frame - start_frame


def write_split_file(split_dir, split_name, sample_names, clip_len, interval, tuple_len):
    split_path = split_dir / f"vcop_{split_name}_{clip_len}_{interval}_{tuple_len}.txt"
    with open(split_path, "w", encoding="utf-8") as f:
        for sample_name in sample_names:
            f.write(sample_name + "\n")
    return split_path


def filter_split(sample_names, root_dir, annotations_root, videos_root, min_frames, strict):
    kept = []
    skipped_missing = []
    skipped_short = []
    skipped_broken = []

    for sample_name in sample_names:
        try:
            annotation = load_annotation(annotations_root, sample_name)
        except Exception as exc:
            if strict:
                raise
            skipped_broken.append(f"{sample_name} (annotation: {exc})")
            continue

        video_path = resolve_video_path(root_dir, videos_root, annotation["video_name"])
        if not video_path.exists():
            if strict:
                raise FileNotFoundError(f"Video file not found: {video_path}")
            skipped_missing.append(annotation["sample_name"])
            continue

        start_sec, end_sec = annotation["clip_time_range"]
        try:
            frame_count = available_window_frames(video_path, start_sec, end_sec)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to probe {video_path}: {exc}")
            skipped_broken.append(f"{annotation['sample_name']} (probe: {exc})")
            continue

        if frame_count < min_frames:
            skipped_short.append(annotation["sample_name"])
            continue

        kept.append(annotation["sample_name"])

    return kept, skipped_missing, skipped_short, skipped_broken


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    annotations_root = resolve_annotations_root(root_dir, args.annotations_dir)
    videos_root = resolve_videos_root(root_dir, args.videos_dir)
    split_dir = root_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(root_dir)
    selected_json_files = manifest.get("selected_json_files", {})
    min_frames = args.clip_len * args.tuple_len + args.interval * (args.tuple_len - 1)

    stats = {}
    for split_name in ("train", "val", "test"):
        sample_names = selected_json_files.get(split_name)
        if sample_names is None:
            print(f"Skip {split_name}: split not found in manifest.json")
            continue

        kept, skipped_missing, skipped_short, skipped_broken = filter_split(
            sample_names,
            root_dir,
            annotations_root,
            videos_root,
            min_frames,
            args.strict,
        )
        split_path = write_split_file(
            split_dir,
            split_name,
            kept,
            args.clip_len,
            args.interval,
            args.tuple_len,
        )
        stats[split_name] = {
            "manifest_entries": len(sample_names),
            "kept": len(kept),
            "missing": len(skipped_missing),
            "short": len(skipped_short),
            "broken": len(skipped_broken),
            "split_path": split_path,
        }

    print("Done.")
    print("root_dir:", root_dir)
    print("annotations_root:", annotations_root)
    print("videos_root:", videos_root)
    print("min_frames:", min_frames)
    for split_name in ("train", "val", "test"):
        split_stats = stats.get(split_name)
        if split_stats is None:
            continue
        print(
            "{}: manifest_entries={}, kept={}, missing={}, short={}, broken={}, split={}".format(
                split_name,
                split_stats["manifest_entries"],
                split_stats["kept"],
                split_stats["missing"],
                split_stats["short"],
                split_stats["broken"],
                split_stats["split_path"],
            )
        )


if __name__ == "__main__":
    main()
