import json
import random
from pathlib import Path

import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

import skvideo.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CSLNewsVCOPDataset(Dataset):
    """
    CSL-News dataset for VCOP using JSON annotations and mp4 videos.

    Expected structure:
        root_dir/
            train_label.json
            val_label.json
            test_label.json
            split/
                vcop_train_16_8_3.txt
                vcop_val_16_8_3.txt
                vcop_test_16_8_3.txt
            rgb/
                *.mp4

    The `videos_dir` argument can also be an absolute path so that JSON
    annotations and the actual video files can live in different directories.
    """

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
        videos_dir="rgb",
    ):
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.fixed_sampling = fixed_sampling
        self.toPIL = transforms.ToPILImage()
        self.split_dir = self.root_dir / "split"
        self.split_name = split_name or ("train" if train else "test")
        self.videos_root = self._resolve_videos_root(videos_dir)
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        self.metadata_by_video = self._load_all_metadata()
        self.samples = self._load_samples(split_file)

    def __len__(self):
        return len(self.samples)

    def _resolve_videos_root(self, videos_dir):
        videos_path = Path(videos_dir)
        if videos_path.is_absolute():
            return videos_path
        return self.root_dir / videos_path

    def _label_path(self, split_name):
        return self.root_dir / f"{split_name}_label.json"

    def _load_label_entries(self, split_name):
        label_path = self._label_path(split_name)
        if not label_path.exists():
            return []

        with open(label_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON list in {label_path}, but got {type(entries).__name__}.")

        normalized_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Expected dict entries in {label_path}, but got {type(entry).__name__}.")
            video_name = entry.get("video")
            if not video_name:
                raise ValueError(f"Missing 'video' field in {label_path}: {entry}")
            normalized_entries.append(entry)
        return normalized_entries

    def _load_all_metadata(self):
        metadata = {}
        for split_name in ("train", "val", "test"):
            for entry in self._load_label_entries(split_name):
                metadata[entry["video"]] = entry
        if not metadata:
            raise FileNotFoundError(
                f"No CSL-News label JSON files were found under {self.root_dir}. "
                "Expected train_label.json / val_label.json / test_label.json."
            )
        return metadata

    def _default_split_path(self):
        return self.split_dir / f"vcop_{self.split_name}_{self.clip_len}_{self.interval}_{self.tuple_len}.txt"

    def _read_split_lines(self, split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _resolve_entry_from_name(self, sample_name):
        entry = self.metadata_by_video.get(sample_name)
        if entry is not None:
            return entry

        basename = Path(sample_name).name
        entry = self.metadata_by_video.get(basename)
        if entry is not None:
            return entry

        raise KeyError(
            f"Sample '{sample_name}' was not found in CSL-News label JSON files under {self.root_dir}."
        )

    def _load_samples(self, split_file):
        if split_file is not None:
            split_path = Path(split_file)
            if split_path.suffix.lower() == ".json":
                with open(split_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                if not isinstance(entries, list):
                    raise ValueError(f"Expected a JSON list in {split_path}.")
                return entries

            sample_names = self._read_split_lines(split_path)
            return [self._resolve_entry_from_name(sample_name) for sample_name in sample_names]

        default_split_path = self._default_split_path()
        if default_split_path.exists():
            sample_names = self._read_split_lines(default_split_path)
            return [self._resolve_entry_from_name(sample_name) for sample_name in sample_names]

        return self._load_label_entries(self.split_name)

    def _resolve_video_path(self, video_name):
        primary_path = self.videos_root / video_name
        if primary_path.exists():
            return primary_path

        fallback_path = self.root_dir / video_name
        if fallback_path.exists():
            return fallback_path

        return primary_path

    def _build_rng(self, idx):
        deterministic = self.fixed_sampling or (not self.train)
        if deterministic:
            return random.Random(idx)
        return random

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample["video"]
        video_path = self._resolve_video_path(video_name)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found for sample '{video_name}'. Checked: {video_path}"
            )

        try:
            videodata = skvideo.io.vread(str(video_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode video '{video_path}': {exc}") from exc

        length = videodata.shape[0]
        if length < self.tuple_total_frames:
            raise ValueError(
                f"Sample {video_name} has only {length} frames, "
                f"but needs at least {self.tuple_total_frames}."
            )

        if self.transforms_ is None:
            raise ValueError("transforms_ must be provided for CSLNewsVCOPDataset")

        rng = self._build_rng(idx)
        tuple_start = rng.randint(0, length - self.tuple_total_frames)

        tuple_clip = []
        tuple_order = list(range(self.tuple_len))
        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
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
