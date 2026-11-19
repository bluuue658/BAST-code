"""PROMISE12 datasets using the released case-level split manifests."""

from collections import OrderedDict
from pathlib import Path, PurePosixPath
import random

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset


def parse_manifest(path):
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(f"Expected image and label paths in {path}: {line!r}")
        entries.append((fields[0], fields[1]))
    if not entries:
        raise ValueError(f"Empty manifest: {path}")
    return entries


def case_id(image_relative_path):
    path = PurePosixPath(image_relative_path)
    if len(path.parts) < 3:
        raise ValueError(f"Cannot obtain case ID from {image_relative_path!r}")
    return path.parent.name


def _read_pair(root, entry):
    image_path, label_path = entry
    image_file = root / image_path
    label_file = root / label_path
    if image_file.suffix.lower() == ".npy":
        image = np.asarray(np.load(image_file), dtype=np.float32)
    else:
        image = np.asarray(Image.open(image_file).convert("L"), dtype=np.float32)
        # Legacy DAA PNG conversion stores images as uint8.
        image = image / 255.0
    if label_file.suffix.lower() == ".npy":
        label = np.asarray(np.load(label_file), dtype=np.uint8)
    else:
        label = np.asarray(Image.open(label_file).convert("L"), dtype=np.uint8)
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image_path}, {label_path}")
    values = set(np.unique(label).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Non-binary label {label_path}: {sorted(values)}")
    return image, label


def _random_rot_flip(image, label):
    turns = np.random.randint(0, 4)
    image = np.rot90(image, turns)
    label = np.rot90(label, turns)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis).copy(), np.flip(label, axis).copy()


def _random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _bcp_augment(image, label, output_size):
    # Matches the official BCP ACDC RandomGenerator probabilities and order.
    if random.random() > 0.5:
        image, label = _random_rot_flip(image, label)
    elif random.random() > 0.5:
        image, label = _random_rotate(image, label)
    height, width = image.shape
    image = zoom(
        image,
        (output_size[0] / height, output_size[1] / width),
        order=0,
    )
    label = zoom(
        label,
        (output_size[0] / height, output_size[1] / width),
        order=0,
    )
    return image.astype(np.float32), label.astype(np.uint8)


class Promise12BCPTrainDataset(Dataset):
    """Labeled slices first, then unlabeled slices, for TwoStreamBatchSampler."""

    def __init__(
        self,
        root,
        labeled_manifest,
        unlabeled_manifest,
        output_size=(256, 256),
    ):
        self.root = Path(root)
        self.labeled_entries = parse_manifest(labeled_manifest)
        self.unlabeled_entries = parse_manifest(unlabeled_manifest)
        labeled_paths = {entry[0] for entry in self.labeled_entries}
        unlabeled_paths = {entry[0] for entry in self.unlabeled_entries}
        if labeled_paths & unlabeled_paths:
            raise ValueError("Labeled and unlabeled slice manifests overlap")
        labeled_cases = {case_id(entry[0]) for entry in self.labeled_entries}
        unlabeled_cases = {case_id(entry[0]) for entry in self.unlabeled_entries}
        if labeled_cases & unlabeled_cases:
            raise ValueError("Labeled and unlabeled case manifests overlap")
        self.entries = self.labeled_entries + self.unlabeled_entries
        self.labeled_count = len(self.labeled_entries)
        self.output_size = tuple(output_size)

    @property
    def labeled_indices(self):
        return list(range(self.labeled_count))

    @property
    def unlabeled_indices(self):
        return list(range(self.labeled_count, len(self.entries)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image, label = _read_pair(self.root, self.entries[index])
        is_labeled = index < self.labeled_count
        if not is_labeled:
            # Prevent accidental access to hidden labels during SSL training.
            label = np.zeros_like(label)
        image, label = _bcp_augment(image, label, self.output_size)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label).long(),
            "is_labeled": torch.tensor(is_labeled),
            "id": self.entries[index][0],
        }


class Promise12VolumeDataset(Dataset):
    """Complete ordered cases for subject-level validation or testing."""

    def __init__(self, root, manifest):
        self.root = Path(root)
        grouped = OrderedDict()
        for entry in parse_manifest(manifest):
            grouped.setdefault(case_id(entry[0]), []).append(entry)
        self.cases = []
        for name, entries in grouped.items():
            entries.sort(key=lambda item: int(Path(item[0]).stem))
            self.cases.append((name, entries))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        name, entries = self.cases[index]
        pairs = [_read_pair(self.root, entry) for entry in entries]
        images = torch.from_numpy(np.stack([pair[0] for pair in pairs])).float()
        labels = torch.from_numpy(np.stack([pair[1] for pair in pairs])).long()
        return {"image": images, "label": labels, "case": name}
