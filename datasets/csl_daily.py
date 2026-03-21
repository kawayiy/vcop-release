# import random
# from pathlib import Path

# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image


# class CSLDailyVCOPDataset(Dataset):
#     """
#     CSL-Daily dataset for VCOP using frame folders.

#     Expected structure:
#         root_dir/
#             sentence/
#                 S000000_P0000_T00/
#                     000000.jpg
#                     000001.jpg
#                     ...
#             split/
#                 vcop_train_16_8_3.txt
#                 vcop_test_16_8_3.txt
#     """

#     def __init__(
#         self,
#         root_dir,
#         clip_len,
#         interval,
#         tuple_len,
#         train=True,
#         transforms_=None,
#         fixed_sampling=False,
#     ):
#         self.root_dir = Path(root_dir)
#         self.sent_dir = self.root_dir / "sentence"
#         self.split_dir = self.root_dir / "split"

#         self.clip_len = clip_len
#         self.interval = interval
#         self.tuple_len = tuple_len
#         self.train = train
#         self.transforms_ = transforms_
#         self.fixed_sampling = fixed_sampling
#         self.toPIL = transforms.ToPILImage()

#         self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

#         if self.train:
#             split_name = f"vcop_train_{clip_len}_{interval}_{tuple_len}.txt"
#         else:
#             split_name = f"vcop_test_{clip_len}_{interval}_{tuple_len}.txt"

#         split_path = self.split_dir / split_name
#         with open(split_path, "r", encoding="utf-8") as f:
#             self.samples = [line.strip() for line in f if line.strip()]

#     def __len__(self):
#         return len(self.samples)

#     def _load_frames(self, sample_name):
#         sample_dir = self.sent_dir / sample_name
#         frame_paths = sorted(sample_dir.glob("*.jpg"))
#         return frame_paths

#     def __getitem__(self, idx):
#         sample_name = self.samples[idx]
#         frame_paths = self._load_frames(sample_name)
#         length = len(frame_paths)

#         tuple_clip = []
#         tuple_order = list(range(self.tuple_len))

#         rng = random.Random(idx) if self.fixed_sampling else random

#         if length < self.tuple_total_frames:
#             raise ValueError(
#                 f"Sample {sample_name} has only {length} frames, "
#                 f"but needs at least {self.tuple_total_frames}."
#             )

#         tuple_start = rng.randint(0, length - self.tuple_total_frames)

#         clip_start = tuple_start
#         for _ in range(self.tuple_len):
#             clip_frame_paths = frame_paths[clip_start: clip_start + self.clip_len]
#             tuple_clip.append(clip_frame_paths)
#             clip_start = clip_start + self.clip_len + self.interval

#         clip_and_order = list(zip(tuple_clip, tuple_order))
#         rng.shuffle(clip_and_order)
#         tuple_clip, tuple_order = zip(*clip_and_order)

#         if self.transforms_ is None:
#             raise ValueError("transforms_ must be provided for CSLDailyVCOPDataset")

#         trans_tuple = []
#         for clip_idx, clip in enumerate(tuple_clip):
#             trans_clip = []

#             if self.fixed_sampling:
#                 clip_rng = random.Random(idx * 100 + clip_idx)
#                 seed = clip_rng.random()
#             else:
#                 seed = random.random()

#             for frame_path in clip:
#                 random.seed(seed)
#                 frame = Image.open(frame_path).convert("RGB")
#                 frame = self.transforms_(frame)
#                 trans_clip.append(frame)

#             trans_clip = torch.stack(trans_clip).permute(1, 0, 2, 3)  # [C, T, H, W]
#             trans_tuple.append(trans_clip)

#         tuple_clip = trans_tuple

#         return torch.stack(tuple_clip), torch.tensor(tuple_order)


import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSLDailyVCOPDataset(Dataset):
    """
    CSL-Daily dataset for VCOP using frame folders.

    Expected structure:
        root_dir/
            sentence/
                S000000_P0000_T00/
                    000000.jpg
                    000001.jpg
                    ...
            split/
                vcop_train_16_8_3.txt
                vcop_test_16_8_3.txt
                vcop_all_16_8_3.txt
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
    ):
        self.root_dir = Path(root_dir)
        self.sent_dir = self.root_dir / "sentence"
        self.split_dir = self.root_dir / "split"

        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.fixed_sampling = fixed_sampling
        self.toPIL = transforms.ToPILImage()

        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if split_file is not None:
            split_path = Path(split_file)
        else:
            if self.train:
                split_name = f"vcop_train_{clip_len}_{interval}_{tuple_len}.txt"
            else:
                split_name = f"vcop_test_{clip_len}_{interval}_{tuple_len}.txt"
            split_path = self.split_dir / split_name

        with open(split_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample_name):
        sample_dir = self.sent_dir / sample_name
        frame_paths = sorted(sample_dir.glob("*.jpg"))
        return frame_paths

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        frame_paths = self._load_frames(sample_name)
        length = len(frame_paths)

        tuple_clip = []
        tuple_order = list(range(self.tuple_len))

        rng = random.Random(idx) if self.fixed_sampling else random

        if length < self.tuple_total_frames:
            raise ValueError(
                f"Sample {sample_name} has only {length} frames, "
                f"but needs at least {self.tuple_total_frames}."
            )

        tuple_start = rng.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip_frame_paths = frame_paths[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip_frame_paths)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        rng.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_ is None:
            raise ValueError("transforms_ must be provided for CSLDailyVCOPDataset")

        trans_tuple = []
        for clip_idx, clip in enumerate(tuple_clip):
            trans_clip = []

            if self.fixed_sampling:
                clip_rng = random.Random(idx * 100 + clip_idx)
                seed = clip_rng.random()
            else:
                seed = random.random()

            for frame_path in clip:
                random.seed(seed)
                frame = Image.open(frame_path).convert("RGB")
                frame = self.transforms_(frame)
                trans_clip.append(frame)

            trans_clip = torch.stack(trans_clip).permute(1, 0, 2, 3)  # [C, T, H, W]
            trans_tuple.append(trans_clip)

        return torch.stack(trans_tuple), torch.tensor(tuple_order)