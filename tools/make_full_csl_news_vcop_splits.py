"""
Example usage:

    python tools/make_full_csl_news_vcop_splits.py

    python tools/make_full_csl_news_vcop_splits.py ^
      --root_dir "D:/path/to/CSL-News" ^
      --videos_dir "D:/path/to/CSL-News/rgb"

    python tools/make_full_csl_news_vcop_splits.py ^
      --root_dir "D:/path/to/CSL-News" ^
      --videos_dir "D:/path/to/CSL-News/rgb" ^
      --clip_len 16 ^
      --interval 8 ^
      --tuple_len 3 ^
      --strict
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

from skvideo.io import ffprobe

DEFAULT_ROOT_DIR = Path(
    r"D:/python_projects/Uni-SLM-data/mini_CSL_News-20260108T160644Z-1-001/mini_CSL_News"
)
DEFAULT_VIDEOS_DIR = "rgb"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CSL-News VCOP split files")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="directory containing *_label.json files",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default=DEFAULT_VIDEOS_DIR,
        help="relative or absolute video directory",
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


def resolve_video_path(root_dir, videos_root, video_name):
    primary_path = videos_root / video_name
    if primary_path.exists():
        return primary_path

    fallback_path = root_dir / video_name
    if fallback_path.exists():
        return fallback_path

    return primary_path


def load_label_entries(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Expected a JSON list in {}".format(label_path))

    valid_entries = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("video"):
            raise ValueError("Invalid label entry in {}: {}".format(label_path, entry))
        valid_entries.append(entry)
    return valid_entries


def estimate_frame_count(video_path):
    metadata = ffprobe(str(video_path)).get("video", {})
    nb_frames = metadata.get("@nb_frames")
    if nb_frames not in (None, "", "N/A"):
        return int(float(nb_frames))

    duration = metadata.get("@duration")
    avg_frame_rate = metadata.get("@avg_frame_rate")
    if duration in (None, "", "N/A") or avg_frame_rate in (None, "", "N/A", "0/0"):
        raise ValueError("Could not infer frame count from ffprobe metadata for {}".format(video_path))

    fps = float(Fraction(avg_frame_rate))
    return int(round(float(duration) * fps))


def filter_split(entries, root_dir, videos_root, min_frames, strict):
    kept = []
    skipped_missing = []
    skipped_short = []
    skipped_broken = []

    for entry in entries:
        video_name = entry["video"]
        video_path = resolve_video_path(root_dir, videos_root, video_name)
        if not video_path.exists():
            if strict:
                raise FileNotFoundError("Video file not found: {}".format(video_path))
            skipped_missing.append(video_name)
            continue

        try:
            frame_count = estimate_frame_count(video_path)
        except Exception as exc:
            if strict:
                raise RuntimeError("Failed to probe {}: {}".format(video_path, exc))
            skipped_broken.append(video_name)
            continue

        if frame_count < min_frames:
            skipped_short.append(video_name)
            continue

        kept.append(video_name)

    return kept, skipped_missing, skipped_short, skipped_broken


def write_split_file(split_dir, split_name, sample_names, clip_len, interval, tuple_len):
    split_path = split_dir / "vcop_{}_{}_{}_{}.txt".format(split_name, clip_len, interval, tuple_len)
    with open(split_path, "w", encoding="utf-8") as f:
        for sample_name in sample_names:
            f.write(sample_name + "\n")
    return split_path


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    videos_root = resolve_videos_root(root_dir, args.videos_dir)
    split_dir = root_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    min_frames = args.clip_len * args.tuple_len + args.interval * (args.tuple_len - 1)

    stats = {}
    for split_name in ("train", "val", "test"):
        label_path = root_dir / "{}_label.json".format(split_name)
        if not label_path.exists():
            print("Skip {}: label file not found at {}".format(split_name, label_path))
            continue

        entries = load_label_entries(label_path)
        kept, skipped_missing, skipped_short, skipped_broken = filter_split(
            entries,
            root_dir,
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
            "label_entries": len(entries),
            "kept": len(kept),
            "missing": len(skipped_missing),
            "short": len(skipped_short),
            "broken": len(skipped_broken),
            "split_path": split_path,
        }

    print("Done.")
    print("root_dir:", root_dir)
    print("videos_root:", videos_root)
    print("min_frames:", min_frames)
    for split_name in ("train", "val", "test"):
        split_stats = stats.get(split_name)
        if split_stats is None:
            continue
        print(
            "{}: labels={}, kept={}, missing={}, short={}, broken={}, split={}".format(
                split_name,
                split_stats["label_entries"],
                split_stats["kept"],
                split_stats["missing"],
                split_stats["short"],
                split_stats["broken"],
                split_stats["split_path"],
            )
        )


if __name__ == "__main__":
    main()
