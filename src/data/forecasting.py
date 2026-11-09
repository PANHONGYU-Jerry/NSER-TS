"""Windowed ETTm1 and Weather forecasting datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


DEFAULT_ROOT = Path("/data/panhongyu/TS-LLM/datasets")


def _resolve_path(name: str, root: Path, path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    if name == "ETTm1":
        return root / "ETT-small" / "ETTm1.csv"
    if name == "Weather":
        return root / "Weather" / "Weather.csv"
    raise ValueError("name must be ETTm1 or Weather")


def _load_storage(name: str, path: Path, normalize: bool) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    values = frame.drop(columns=["date"], errors="ignore").select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if "date" in frame:
        date = pd.to_datetime(frame["date"])
        hour = date.dt.hour.to_numpy(dtype=np.float32)
        weekday = date.dt.dayofweek.to_numpy(dtype=np.float32)
        month = date.dt.month.to_numpy(dtype=np.float32) - 1.0
        time_features = np.stack(
            [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * weekday / 7),
                np.cos(2 * np.pi * weekday / 7),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
            ],
            axis=-1,
        ).astype(np.float32)
    else:
        time_features = np.zeros((len(values), 6), dtype=np.float32)
    total = len(values)
    if name == "ETTm1":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
        if total < test_end:
            raise ValueError(f"ETTm1 needs at least {test_end} rows, found {total}")
        borders = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}
    else:
        train_end, val_end = int(total * 0.7), int(total * 0.8)
        borders = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, total)}
    if normalize:
        mean = values[:train_end].mean(axis=0, keepdims=True)
        std = values[:train_end].std(axis=0, keepdims=True)
        values = (values - mean) / np.maximum(std, 1e-6)
    return {
        "values": values,
        "time_features": time_features,
        "borders": borders,
        "columns": [c for c in frame.columns if c != "date"],
    }


class ForecastingDataset(Dataset[TimeSeriesSample]):
    """View windows over one shared matrix; windows are not pre-materialized."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        path: str | Path | None = None,
        seq_len: int = 96,
        pred_len: int = 96,
        stride: int = 1,
        normalize: bool = True,
        _storage: dict[str, object] | None = None,
    ) -> None:
        if name not in ("ETTm1", "Weather"):
            raise ValueError("name must be ETTm1 or Weather")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.name, self.split = name, split
        self.seq_len, self.pred_len, self.stride = int(seq_len), int(pred_len), int(stride)
        if min(self.seq_len, self.pred_len, self.stride) <= 0:
            raise ValueError("seq_len, pred_len, and stride must be positive")
        storage = _storage or _load_storage(name, _resolve_path(name, Path(root), path), normalize)
        self.values = storage["values"]
        self.time_features = storage["time_features"]
        section_start, section_end = storage["borders"][split]
        self.first_start = section_start if split == "train" else section_start - self.seq_len
        self.last_start = section_end - self.seq_len - self.pred_len
        self.n_windows = max(0, (self.last_start - self.first_start) // self.stride + 1)
        self.channel_names = storage["columns"]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> TimeSeriesSample:
        if not 0 <= index < self.n_windows:
            raise IndexError(index)
        start = self.first_start + index * self.stride
        middle, end = start + self.seq_len, start + self.seq_len + self.pred_len
        x = self.values[start:middle].T
        future = torch.from_numpy(np.asarray(self.values[middle:end].T, dtype=np.float32))
        return make_sample(
            x,
            future,
            f"forecast:{self.name}:{self.split}:{start}",
            time_features=self.time_features[start:middle],
        )


def make_forecasting_datasets(
    name: str,
    root: str | Path = DEFAULT_ROOT,
    path: str | Path | None = None,
    seq_len: int = 96,
    pred_len: int = 96,
    stride: int = 1,
    normalize: bool = True,
) -> dict[str, ForecastingDataset]:
    storage = _load_storage(name, _resolve_path(name, Path(root), path), normalize)
    return {
        split: ForecastingDataset(
            name=name,
            split=split,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            _storage=storage,
        )
        for split in ("train", "val", "test")
    }


def make_forecasting(name: str, split: str = "train", **kwargs: object) -> ForecastingDataset:
    return ForecastingDataset(name=name, split=split, **kwargs)
