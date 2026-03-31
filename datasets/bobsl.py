import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BOBSLVCOPDataset(Dataset):
    """
    BOBSL dataset for VCOP using VIA JSON annotations and mp4 videos.

    Expected structure:
        root_dir/
            manifest.json
            manual_annotations/
                continuous_sign_sequences/
                    cslr-json-v2/
                        *.json
            original_data/
                videos/
                    mp4/
                        *.mp4
            split/
                vcop_train_16_8_3.txt
                vcop_val_16_8_3.txt
                vcop_test_16_8_3.txt

    Split files list annotation JSON names, one per line.
    """

    DEFAULT_ANNOTATIONS_DIR = "manual_annotations/continuous_sign_sequences/cslr-json-v2"

    def __init__(
        self,
        root_dir,
        clip_len,
        interval,
        tuple_len,
        train=True,
        transforms_=None,
        fixed_sampling=False,
        split_file=None,
        split_name=None,
        videos_dir="original_data/videos/mp4",
        annotations_dir=DEFAULT_ANNOTATIONS_DIR,
    ):
        self.root_dir = Path(root_dir)
        self.annotations_dir = self._resolve_dataset_path(annotations_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.split_dir = self.root_dir / "split"

        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.fixed_sampling = fixed_sampling
        self.toPIL = transforms.ToPILImage()
        self.split_name = split_name or ("train" if train else "test")
        self.videos_root = self._resolve_videos_root(videos_dir)
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        self._annotation_cache = {}
        self._video_info_cache = {}
        self.samples = self._load_samples(split_file)

    def __len__(self):
        return len(self.samples)

    def _resolve_videos_root(self, videos_dir):
        videos_path = Path(videos_dir)
        if videos_path.is_absolute():
            return videos_path
        return self.root_dir / videos_path

    def _resolve_dataset_path(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _default_split_path(self):
        return self.split_dir / f"vcop_{self.split_name}_{self.clip_len}_{self.interval}_{self.tuple_len}.txt"

    def _normalize_sample_name(self, sample_name):
        normalized_name = Path(sample_name).name.strip()
        if not normalized_name:
            raise ValueError("Encountered an empty BOBSL sample name")
        if not normalized_name.endswith(".json"):
            normalized_name += ".json"
        return normalized_name

    def _read_split_lines(self, split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return [self._normalize_sample_name(line) for line in f if line.strip()]

    def _load_manifest_split(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found at {self.manifest_path}. "
                "Provide --train_split/--val_split or add manifest.json."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        selected_json_files = manifest.get("selected_json_files", {})
        sample_names = selected_json_files.get(self.split_name)
        if sample_names is None:
            raise KeyError(
                f"Split '{self.split_name}' was not found in {self.manifest_path}. "
                "Expected one of: train, val, test."
            )

        return [self._normalize_sample_name(sample_name) for sample_name in sample_names]

    def _load_samples(self, split_file):
        if split_file is not None:
            return self._read_split_lines(Path(split_file))

        default_split_path = self._default_split_path()
        if default_split_path.exists():
            return self._read_split_lines(default_split_path)

        return self._load_manifest_split()

    def _sample_json_path(self, sample_name):
        return self.annotations_dir / self._normalize_sample_name(sample_name)

    def _parse_time_range(self, src):
        if not src or "#t=" not in src:
            raise ValueError(f"Expected '#t=start,end' in BOBSL src field, but got: {src}")

        time_range = src.rsplit("#t=", 1)[1]
        start_str, end_str = time_range.split(",", 1)
        start_sec = float(start_str)
        end_sec = float(end_str)
        if end_sec <= start_sec:
            raise ValueError(f"Invalid time range in BOBSL src field: {src}")
        return start_sec, end_sec

    def _load_annotation(self, sample_name):
        normalized_name = self._normalize_sample_name(sample_name)
        cached = self._annotation_cache.get(normalized_name)
        if cached is not None:
            return cached

        sample_path = self._sample_json_path(normalized_name)
        if not sample_path.exists():
            raise FileNotFoundError(f"BOBSL annotation file not found: {sample_path}")

        with open(sample_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        file_entries = payload.get("file", {})
        if not file_entries:
            raise ValueError(f"No video file entries found in BOBSL annotation: {sample_path}")

        first_file_entry = next(iter(file_entries.values()))
        video_name = first_file_entry.get("fname")
        if not video_name:
            raise ValueError(f"Missing fname in BOBSL annotation: {sample_path}")

        src = first_file_entry.get("src")
        start_sec, end_sec = self._parse_time_range(src)

        annotation = {
            "sample_name": normalized_name,
            "sample_id": payload.get("project", {}).get("pid", normalized_name[:-5]),
            "video_name": Path(video_name).name,
            "clip_time_range": (start_sec, end_sec),
        }
        self._annotation_cache[normalized_name] = annotation
        return annotation

    def _resolve_video_path(self, video_name):
        primary_path = self.videos_root / video_name
        if primary_path.exists():
            return primary_path

        fallback_path = self.root_dir / video_name
        if fallback_path.exists():
            return fallback_path

        return primary_path

    def _probe_video_info(self, video_path):
        cache_key = str(video_path)
        cached = self._video_info_cache.get(cache_key)
        if cached is not None:
            return cached

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

        info = {
            "fps": fps,
            "total_frames": total_frames,
        }
        self._video_info_cache[cache_key] = info
        return info

    def _read_frame_window(self, video_path, start_frame, end_frame):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        capture.release()
        if not frames:
            raise RuntimeError(
                f"Failed to decode frames {start_frame}:{end_frame} from BOBSL video {video_path}"
            )
        return np.stack(frames)

    def _build_rng(self, idx):
        deterministic = self.fixed_sampling or (not self.train)
        if deterministic:
            return random.Random(idx)
        return random

    def __getitem__(self, idx):
        sample = self._load_annotation(self.samples[idx])
        video_name = sample["video_name"]
        video_path = self._resolve_video_path(video_name)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found for BOBSL sample '{sample['sample_name']}'. Checked: {video_path}"
            )

        if self.transforms_ is None:
            raise ValueError("transforms_ must be provided for BOBSLVCOPDataset")

        video_info = self._probe_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        start_sec, end_sec = sample["clip_time_range"]

        start_frame = max(0, int(math.floor(start_sec * fps)))
        end_frame = min(total_frames, int(math.ceil(end_sec * fps)))
        available_frames = end_frame - start_frame
        if available_frames < self.tuple_total_frames:
            raise ValueError(
                f"BOBSL sample {sample['sample_name']} has only {available_frames} frames "
                f"inside {start_sec:.3f}-{end_sec:.3f}s, but needs at least {self.tuple_total_frames}."
            )

        try:
            clip_data = self._read_frame_window(video_path, start_frame, end_frame)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode BOBSL time window from '{video_path}': {exc}") from exc

        rng = self._build_rng(idx)
        tuple_start = rng.randint(0, available_frames - self.tuple_total_frames)

        tuple_clip = []
        tuple_order = list(range(self.tuple_len))
        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = clip_data[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        rng.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        deterministic = self.fixed_sampling or (not self.train)
        trans_tuple = []
        for clip_idx, clip in enumerate(tuple_clip):
            trans_clip = []
            clip_rng = random.Random(idx * 100 + clip_idx) if deterministic else random
            seed = clip_rng.random()

            for frame in clip:
                random.seed(seed)
                frame = self.toPIL(frame)
                frame = self.transforms_(frame)
                trans_clip.append(frame)

            trans_tuple.append(torch.stack(trans_clip).permute(1, 0, 2, 3))

        return torch.stack(trans_tuple), torch.tensor(tuple_order)
