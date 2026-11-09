"""Adapters for univariate UCR morphology / shape classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


DEFAULT_ROOT = Path("/data/panhongyu/TS-LLM/datasets")
# Class label = waveform / contour / spectrum morphology (not activity events).
UCR_DATASETS = (
    "CBF",
    "ECG5000",
    "FordA",
    "FordB",
    "ElectricDevices",
    "StarLightCurves",
    "SyntheticControl",
    "TwoLeadECG",
    "TwoPatterns",
    "Wafer",
    "Coffee",
    "ItalyPowerDemand",
    "Plane",
    "Trace",
    "MoteStrain",
)


def _read_official_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return values[:, 1:], values[:, 0]


def _stratified_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = min(max(1, int(round(len(idx) * val_ratio))), max(1, len(idx) - 1))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class UCRDataset(Dataset[TimeSeriesSample]):
    """One official UCR split, with validation carved only from train."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalize: bool = False,
    ) -> None:
        if name not in UCR_DATASETS:
            raise ValueError(f"Unknown UCR dataset {name!r}; choices: {UCR_DATASETS}")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.name, self.split = name, split
        dataset_root = Path(root) / name
        source_split = "TEST" if split == "test" else "TRAIN"
        x, raw_y = _read_official_file(dataset_root / f"{name}_{source_split}.txt")

        train_labels = _read_official_file(dataset_root / f"{name}_TRAIN.txt")[1]
        test_labels = _read_official_file(dataset_root / f"{name}_TEST.txt")[1]
        classes = sorted(set(train_labels.tolist()) | set(test_labels.tolist()))
        self.class_to_index = {label: i for i, label in enumerate(classes)}
        y = np.asarray([self.class_to_index[float(label)] for label in raw_y], dtype=np.int64)

        if split != "test":
            train_idx, val_idx = _stratified_indices(raw_y, val_ratio, seed)
            selected = train_idx if split == "train" else val_idx
            x, y = x[selected], y[selected]
            self.source_indices = selected
        else:
            self.source_indices = np.arange(len(y))
        if normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / np.maximum(std, 1e-6)
        self.x = np.asarray(x, dtype=np.float32)
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> TimeSeriesSample:
        source = int(self.source_indices[index])
        return make_sample(
            self.x[index][None, :],
            torch.tensor(int(self.y[index]), dtype=torch.long),
            f"ucr:{self.name}:{self.split}:{source}",
        )


def make_ucr_datasets(
    name: str,
    root: str | Path = DEFAULT_ROOT,
    val_ratio: float = 0.2,
    seed: int = 42,
    normalize: bool = False,
) -> dict[str, UCRDataset]:
    common = dict(name=name, root=root, val_ratio=val_ratio, seed=seed, normalize=normalize)
    return {split: UCRDataset(split=split, **common) for split in ("train", "val", "test")}


def make_ucr(name: str, split: str = "train", **kwargs: object) -> UCRDataset:
    return UCRDataset(name=name, split=split, **kwargs)
