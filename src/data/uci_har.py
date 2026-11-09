"""UCI HAR raw-signal adapter with subject-disjoint validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


DEFAULT_ROOT = Path("/data/panhongyu/TS-LLM/datasets/UCI_HAR")
RAW_CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


def _load_part(root: Path, part: str) -> dict[str, np.ndarray]:
    signal_root = root / part / "Inertial Signals"
    channels = [
        np.loadtxt(signal_root / f"{name}_{part}.txt", dtype=np.float32)
        for name in RAW_CHANNELS
    ]
    return {
        "x": np.stack(channels, axis=1),  # [N,C,T]
        "y": np.loadtxt(root / part / f"y_{part}.txt", dtype=np.int64) - 1,
        "subject": np.loadtxt(root / part / f"subject_{part}.txt", dtype=np.int64),
    }


def _subject_indices(subjects: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = min(max(1, int(round(len(unique) * val_ratio))), len(unique) - 1)
    val_subjects = unique[:n_val]
    return np.flatnonzero(~np.isin(subjects, val_subjects)), np.flatnonzero(np.isin(subjects, val_subjects))


def _load_storage(root: Path, val_ratio: float, seed: int, normalize: bool) -> dict[str, object]:
    if not root.exists():
        raise FileNotFoundError(root)
    train, test = _load_part(root, "train"), _load_part(root, "test")
    train_idx, val_idx = _subject_indices(train["subject"], val_ratio, seed)
    if normalize:
        reference = train["x"][train_idx]
        mean = reference.mean(axis=(0, 2), keepdims=True)
        std = reference.std(axis=(0, 2), keepdims=True)
        train["x"] = (train["x"] - mean) / np.maximum(std, 1e-6)
        test["x"] = (test["x"] - mean) / np.maximum(std, 1e-6)
    return {"train_data": train, "test_data": test, "train_idx": train_idx, "val_idx": val_idx}


class UCIHARDataset(Dataset[TimeSeriesSample]):
    def __init__(
        self,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalize: bool = True,
        _storage: dict[str, object] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        storage = _storage or _load_storage(Path(root), val_ratio, seed, normalize)
        if split == "test":
            self.data = storage["test_data"]
            self.indices = np.arange(len(self.data["y"]), dtype=np.int64)
        else:
            self.data = storage["train_data"]
            self.indices = storage[f"{split}_idx"]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def y(self) -> np.ndarray:
        # Expose split-local labels so --label-fraction / --eval-fraction
        # (and any code that expects dataset.y like UCR/SMD) work on HAR.
        return np.asarray(self.data["y"])[self.indices]

    def __getitem__(self, index: int) -> TimeSeriesSample:
        source = int(self.indices[index])
        subject = int(self.data["subject"][source])
        return make_sample(
            self.data["x"][source],
            torch.tensor(int(self.data["y"][source]), dtype=torch.long),
            f"uci_har:{self.split}:subject{subject}:window{source}",
        )


def make_uci_har_datasets(
    root: str | Path = DEFAULT_ROOT,
    val_ratio: float = 0.2,
    seed: int = 42,
    normalize: bool = True,
) -> dict[str, UCIHARDataset]:
    storage = _load_storage(Path(root), val_ratio, seed, normalize)
    return {
        split: UCIHARDataset(split=split, _storage=storage)
        for split in ("train", "val", "test")
    }


def make_uci_har(split: str = "train", **kwargs: object) -> UCIHARDataset:
    return UCIHARDataset(split=split, **kwargs)
