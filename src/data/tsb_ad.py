"""Lazy, one-CSV-per-sample adapter for TSB-AD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


DEFAULT_ROOT = Path("/data/panhongyu/TS-LLM/datasets")
_NAME_RE = re.compile(r"_tr_(?P<train>\d+)_1st_(?P<first>\d+)\.csv$")


def _metadata(filename: str) -> tuple[int, int]:
    match = _NAME_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse train prefix from {filename}")
    return int(match.group("train")), int(match.group("first"))


def _anomaly_events(labels: np.ndarray) -> torch.Tensor:
    active = labels.astype(bool)
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    length = max(1, len(labels))
    return torch.tensor(
        [[start / length, end / length] for start, end in zip(starts, ends)],
        dtype=torch.float32,
    ).reshape(-1, 2)


def standardize_train_prefix(
    features: np.ndarray, train_end: int, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler fit on the normal train prefix only, applied to the full series."""

    prefix = features[: max(int(train_end), 1)]
    mean = prefix.mean(axis=0, keepdims=True).astype(np.float32)
    std = prefix.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return (features - mean) / std, mean, std


def _select_channels(
    features: np.ndarray, train_end: int, max_channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most ``max_channels`` by train-prefix variance (stable order)."""
    channel_count = features.shape[1]
    if channel_count <= max_channels:
        return features, np.arange(channel_count, dtype=np.int64)
    prefix = features[: max(int(train_end), 1)]
    variances = prefix.var(axis=0)
    chosen = np.sort(np.argsort(-variances)[:max_channels].astype(np.int64))
    return features[:, chosen], chosen


def split_tsb_ad_tuning_files(
    modality: str = "M",
    root: str | Path = DEFAULT_ROOT,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    skip_missing: bool = True,
) -> tuple[list[str], list[str]]:
    """Split tuning series by file so train/validation windows never overlap."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    modality = modality.upper()
    root = Path(root)
    listing = root / "TSB-AD" / f"TSB-AD-{modality}-Tuning.csv"
    names = pd.read_csv(listing)["file_name"].astype(str).tolist()
    if skip_missing:
        series_root = root / f"TSB-AD-{modality}"
        names = [name for name in names if (series_root / name).exists()]
    if len(names) < 2:
        raise ValueError("TSB-AD tuning split needs at least two present series")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_val = min(max(1, int(round(len(names) * val_ratio))), len(names) - 1)
    validation = [names[index] for index in order[:n_val]]
    train = [names[index] for index in order[n_val:]]
    return train, validation


class TSBADDataset(Dataset[TimeSeriesSample]):
    """Lazy windowed TSB-AD dataset.

    Long benchmark series cannot be passed directly to quadratic-attention
    encoders.  Each index therefore reads one non-overlapping (by default)
    window while preserving point labels and window-local event boundaries.
    """

    def __init__(
        self,
        modality: str = "M",
        split: str = "tuning",
        root: str | Path = DEFAULT_ROOT,
        split_file: str | Path | None = None,
        series_root: str | Path | None = None,
        skip_missing: bool = True,
        window_size: int = 512,
        stride: int | None = None,
        max_channels: int = 64,
        filenames: Sequence[str] | None = None,
        normal_prefix_only: bool = False,
        include_anomaly: bool = False,
        normalize: bool = True,
    ) -> None:
        modality = modality.upper()
        split = split.lower()
        if modality not in ("U", "M"):
            raise ValueError("modality must be U or M")
        if split not in ("tuning", "eva"):
            raise ValueError("split must be tuning or eva")
        if int(max_channels) <= 0:
            raise ValueError("max_channels must be positive")
        self.modality, self.split = modality, split
        self.max_channels = int(max_channels)
        self.normal_prefix_only = bool(normal_prefix_only)
        self.include_anomaly = bool(include_anomaly)
        self.normalize = bool(normalize)
        self._file_cache: dict[str, dict[str, np.ndarray]] = {}
        root = Path(root)
        self.series_root = Path(series_root) if series_root else root / f"TSB-AD-{modality}"
        listing = Path(split_file) if split_file else root / "TSB-AD" / f"TSB-AD-{modality}-{split.title()}.csv"
        names = (
            list(filenames)
            if filenames is not None
            else pd.read_csv(listing)["file_name"].astype(str).tolist()
        )
        self.filenames = [name for name in names if (self.series_root / name).exists()] if skip_missing else names
        self.window_size = int(window_size)
        self.stride = int(stride or window_size)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.windows: list[tuple[str, int, int]] = []
        self.train_indices = {
            name: _metadata(name)[0] for name in self.filenames
        }
        for name in self.filenames:
            path = self.series_root / name
            length = len(pd.read_csv(path, usecols=["Label"]))
            if self.normal_prefix_only and not self.include_anomaly:
                train_index = self.train_indices[name]
                length = min(length, max(train_index, 1))
            if length <= self.window_size:
                self.windows.append((name, 0, length))
                continue
            starts = list(range(0, length - self.window_size + 1, self.stride))
            if starts[-1] + self.window_size < length:
                starts.append(length - self.window_size)
            self.windows.extend(
                (name, start, min(start + self.window_size, length)) for start in starts
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _window_labels(self, index: int) -> np.ndarray:
        filename, window_start, window_end = self.windows[index]
        stored = self._load_file(filename)
        return stored["labels"][window_start:window_end]

    def anomaly_weights(self) -> np.ndarray:
        """Per-window sampling weights for balanced normal/anomaly training.

        Windows containing at least one labelled anomaly point are weighted so
        that, on average, half of every training batch is anomalous — anomaly
        windows are rare in ``normal_prefix_only`` corpora and otherwise drown
        in normal examples.
        """
        has_anomaly = np.asarray(
            [bool(self._window_labels(index).astype(bool).any()) for index in range(len(self))]
        )
        n_anomaly = int(has_anomaly.sum())
        n_normal = len(has_anomaly) - n_anomaly
        if n_anomaly == 0 or n_normal == 0:
            return np.ones(len(has_anomaly), dtype=np.float64)
        weights = np.where(has_anomaly, 1.0 / n_anomaly, 1.0 / n_normal)
        return weights

    def _load_file(self, filename: str) -> dict[str, np.ndarray]:
        cached = self._file_cache.get(filename)
        if cached is not None:
            return cached
        path = self.series_root / filename
        frame = pd.read_csv(path).dropna()
        if "Label" not in frame:
            raise ValueError(f"Missing Label column: {path}")
        labels = frame["Label"].to_numpy(dtype=np.int64, copy=True)
        features = (
            frame.drop(columns=["Label"])
            .select_dtypes(include=[np.number])
            .to_numpy(dtype=np.float32, copy=True)
        )
        if features.shape[1] == 0:
            raise ValueError(f"No numeric signal columns: {path}")
        train_index, first_anomaly = _metadata(filename)
        train_index = max(1, min(int(train_index), len(features)))
        features, channel_indices = _select_channels(
            features, train_index, self.max_channels
        )
        if self.normalize:
            features, _, _ = standardize_train_prefix(features, train_index)
        n_real = int(features.shape[1])
        if n_real < self.max_channels:
            pad = np.zeros(
                (features.shape[0], self.max_channels - n_real), dtype=np.float32
            )
            features = np.concatenate([features, pad], axis=1)
        cached = {
            "features": features,
            "labels": labels,
            "channel_indices": channel_indices,
            "n_real_channels": np.int64(n_real),
            "train_index": np.int64(train_index),
            "first_anomaly": np.int64(first_anomaly),
        }
        self._file_cache[filename] = cached
        return cached

    def __getitem__(self, index: int) -> TimeSeriesSample:
        filename, window_start, window_end = self.windows[index]
        stored = self._load_file(filename)
        features = stored["features"][window_start:window_end]
        labels = stored["labels"][window_start:window_end]
        train_index = int(stored["train_index"])
        first_anomaly = int(stored["first_anomaly"])
        n_real = int(stored["n_real_channels"])
        local_train_end = min(max(train_index - window_start, 0), len(features))
        train_mask = torch.arange(len(features)) < local_train_end
        task_label = {
            "point_labels": torch.from_numpy(labels),
            "train_prefix_mask": train_mask,
            "train_index": local_train_end,
            "global_train_index": train_index,
            "first_anomaly": first_anomaly - window_start,
            "window_start": window_start,
            "window_end": window_end,
            "modality": self.modality,
            "split": self.split,
            "n_real_channels": n_real,
            "channel_indices": torch.from_numpy(np.asarray(stored["channel_indices"])),
        }
        sample = make_sample(
            features.T,
            task_label,
            f"tsb_ad:{self.modality}:{self.split}:{filename}:{window_start}:{window_end}",
            events_gt=_anomaly_events(labels),
        )
        sample.channel_mask[:] = False
        sample.channel_mask[:n_real] = True
        return sample


def make_tsb_ad(**kwargs: object) -> TSBADDataset:
    return TSBADDataset(**kwargs)
