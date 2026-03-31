import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PhoenixVCOPDataset(Dataset):
    """
    PHOENIX-2014-T dataset for VCOP using frame folders.

    Expected structure:
        root_dir/
            annotations/
                manual/
                    PHOENIX-2014-T.train.corpus.csv
                    PHOENIX-2014-T.dev.corpus.csv
                    PHOENIX-2014-T.test.corpus.csv
            features/
                fullFrame-210x260px/
                    train/
                    dev/
                    test/
            split/
                vcop_train_16_8_3.txt
                vcop_val_16_8_3.txt
                vcop_test_16_8_3.txt
    """

    DEFAULT_FEATURES_DIR = "features/fullFrame-210x260px"
    DEFAULT_ANNOTATIONS_DIR = "annotations/manual"

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
        features_dir=DEFAULT_FEATURES_DIR,
        annotations_dir=DEFAULT_ANNOTATIONS_DIR,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_root = self._resolve_dataset_root(self.root_dir)
        self.features_root = self._resolve_dataset_path(features_dir)
        self.annotations_root = self._resolve_dataset_path(annotations_dir)
        self.split_dir = self.dataset_root / "split"

        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.fixed_sampling = fixed_sampling
        self.toPIL = transforms.ToPILImage()
        self.requested_split_name = split_name or ("train" if train else "test")
        self.canonical_split_name = self._canonical_split_name(self.requested_split_name)
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        self.metadata_by_name = self._load_all_metadata()

        self.samples = self._load_samples(split_file)
    def __len__(self):
        return len(self.samples)

    def _resolve_dataset_root(self, root_dir):
        direct_features = root_dir / self.DEFAULT_FEATURES_DIR
        direct_annotations = root_dir / self.DEFAULT_ANNOTATIONS_DIR
        if direct_features.exists() and direct_annotations.exists():
            return root_dir

        nested_root = root_dir / "PHOENIX-2014-T"
        nested_features = nested_root / self.DEFAULT_FEATURES_DIR
        nested_annotations = nested_root / self.DEFAULT_ANNOTATIONS_DIR
        if nested_features.exists() and nested_annotations.exists():
            return nested_root

        return root_dir

    def _resolve_dataset_path(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.dataset_root / path

    def _canonical_split_name(self, split_name):
        if split_name == "val":
            return "dev"
        return split_name

    def _annotation_csv_path(self, split_name):
        return self.annotations_root / f"PHOENIX-2014-T.{split_name}.corpus.csv"

    def _load_annotation_rows(self, split_name):
        csv_path = self._annotation_csv_path(split_name)
        if not csv_path.exists():
            return []

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

    def _load_all_metadata(self):
        metadata = {}
        for split_name in ("train", "dev", "test"):
            for row in self._load_annotation_rows(split_name):
                metadata[row["name"]] = row

        if not metadata:
            raise FileNotFoundError(
                f"No PHOENIX annotation CSV files were found under {self.annotations_root}. "
                "Expected PHOENIX-2014-T.train/dev/test.corpus.csv."
            )
        return metadata

    def _default_split_path(self):
        return self.split_dir / f"vcop_{self.requested_split_name}_{self.clip_len}_{self.interval}_{self.tuple_len}.txt"

    def _read_split_lines(self, split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _resolve_entry_from_name(self, sample_name):
        entry = self.metadata_by_name.get(sample_name)
        if entry is not None:
            return entry

        basename = Path(sample_name).name
        entry = self.metadata_by_name.get(basename)
        if entry is not None:
            return entry

        raise KeyError(
            f"Sample '{sample_name}' was not found in PHOENIX annotation CSV files under {self.annotations_root}."
        )

    def _load_samples(self, split_file):
        if split_file is not None:
            sample_names = self._read_split_lines(Path(split_file))
            return [self._resolve_entry_from_name(sample_name) for sample_name in sample_names]

        default_split_path = self._default_split_path()
        if default_split_path.exists():
            sample_names = self._read_split_lines(default_split_path)
            return [self._resolve_entry_from_name(sample_name) for sample_name in sample_names]

        return self._load_annotation_rows(self.canonical_split_name)

    def _load_frames(self, sample):
        sample_dir = self.features_root / sample["split"] / sample["name"]
        if not sample_dir.exists():
            raise FileNotFoundError(f"PHOENIX frame folder not found: {sample_dir}")

        frame_paths = sorted(sample_dir.glob("*.png"))
        if not frame_paths:
            frame_paths = sorted(sample_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(sample_dir.glob("*.jpeg"))
        return frame_paths

    def _build_rng(self, idx):
        deterministic = self.fixed_sampling or (not self.train)
        if deterministic:
            return random.Random(idx)
        return random

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = self._load_frames(sample)
        length = len(frame_paths)

        if length < self.tuple_total_frames:
            raise ValueError(
                f"Sample {sample['name']} has only {length} frames, "
                f"but needs at least {self.tuple_total_frames}."
            )

        if self.transforms_ is None:
            raise ValueError("transforms_ must be provided for PhoenixVCOPDataset")

        rng = self._build_rng(idx)
        tuple_start = rng.randint(0, length - self.tuple_total_frames)

        tuple_clip = []
        tuple_order = list(range(self.tuple_len))
        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip_frame_paths = frame_paths[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip_frame_paths)
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

            for frame_path in clip:
                random.seed(seed)
                frame = Image.open(frame_path).convert("RGB")
                frame = self.transforms_(frame)
                trans_clip.append(frame)

            trans_tuple.append(torch.stack(trans_clip).permute(1, 0, 2, 3))

        return torch.stack(trans_tuple), torch.tensor(tuple_order)
