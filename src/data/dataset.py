"""
Flickr30K dataset loader.

Responsibilities
----------------
1. Parse captions.txt  →  {image_id: [caption1, …, caption5]}
2. Build image_id → absolute image path mapping
3. Create a deterministic train / test split at the IMAGE level
4. Provide PyTorch Dataset classes for efficient batch loading.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from PIL import Image
import torch
from torch.utils.data import Dataset

import config as cfg


class Flickr30kDataset:
    """Flickr30K core dataset manager with caption mapping and splits."""

    def __init__(
        self,
        images_dir: str = cfg.FLICKR30K_DIR,
        captions_file: str = cfg.FLICKR30K_CAPTIONS_FILE,
        test_ratio: float = cfg.TEST_SPLIT_RATIO,
        seed: int = cfg.RANDOM_SEED,
    ):
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.test_ratio = test_ratio
        self.seed = seed

        # Core data structures
        self.image_ids: List[str] = []                       # sorted list of unique image IDs
        self.id_to_path: Dict[str, str] = {}                 # image_id → absolute path
        self.id_to_captions: Dict[str, List[str]] = defaultdict(list)  # image_id → captions

        # Split indices
        self.train_ids: List[str] = []
        self.test_ids: List[str] = []

        self._load_captions()
        self._build_image_paths()
        self._build_split()

    def _load_captions(self) -> None:
        """Parse captions.txt (CSV with header: image,caption)."""
        with open(self.captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                image_filename = row[0].strip()
                caption = ",".join(row[1:]).strip()
                self.id_to_captions[image_filename].append(caption)

        self.image_ids = sorted(self.id_to_captions.keys())

    def _build_image_paths(self) -> None:
        for img_id in self.image_ids:
            self.id_to_path[img_id] = os.path.join(self.images_dir, img_id)

    def _build_split(self) -> None:
        """Deterministic train/test split at IMAGE level."""
        rng = random.Random(self.seed)
        ids = list(self.image_ids)
        rng.shuffle(ids)

        n_test = int(len(ids) * self.test_ratio)
        self.test_ids = sorted(ids[:n_test])
        self.train_ids = sorted(ids[n_test:])

    def get_all_image_ids(self) -> List[str]:
        return list(self.image_ids)

    def get_image_path(self, image_id: str) -> str:
        return self.id_to_path[image_id]

    def get_captions(self, image_id: str) -> List[str]:
        return self.id_to_captions[image_id]

    def get_split(self, split: str = "train") -> List[str]:
        if split == "train":
            return list(self.train_ids)
        elif split == "test":
            return list(self.test_ids)
        else:
            raise ValueError(f"Unknown split: {split!r}")


class Flickr30kImageDataset(Dataset):
    """
    PyTorch Dataset for efficient image loading.
    Used for offline embedding generation.
    """
    def __init__(self, core_dataset: Flickr30kDataset, split: str = "all"):
        self.core_dataset = core_dataset
        if split == "all":
            self.image_ids = self.core_dataset.get_all_image_ids()
        else:
            self.image_ids = self.core_dataset.get_split(split)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        img_path = self.core_dataset.get_image_path(img_id)
        
        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback if image is corrupt (rare in Flickr30k but standard practice to handle)
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        return {
            "image_id": img_id,
            "image": image,
            "index": idx
        }
