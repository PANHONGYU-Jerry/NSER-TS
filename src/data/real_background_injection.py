"""Inject controlled events into real training backgrounds."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample
from .controlled_scm import EVENT_TO_ID, EVENT_TYPES, _event_effect
from .forecasting import ForecastingDataset, make_forecasting_datasets
from .tsb_ad import TSBADDataset, split_tsb_ad_tuning_files
from .uci_har import UCIHARDataset
from .ucr import UCRDataset


def _fit_window_length(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Crop or linearly resample ``[C,T]`` to the requested length."""

    channels, length = values.shape
    if length == seq_len:
        return values
    if length > seq_len:
        return values[:, :seq_len]
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    target = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    resized = np.stack(
        [np.interp(target, x, values[channel]) for channel in range(channels)],
        axis=0,
    )
    return resized.astype(np.float32)


class RealBackgroundInjectionDataset(Dataset[TimeSeriesSample]):
    """Wrap real train windows and inject one or more synthetic events."""

    BACKGROUNDS = ("ETTm1", "Weather", "TSB-AD", "UCR", "UCI_HAR")

    def __init__(
        self,
        *,
        size: int = 4000,
        root: str = "/data/panhongyu/TS-LLM/datasets",
        seed: int = 42,
        seq_len: int = 128,
        max_channels: int = 64,
        sources: Optional[Sequence[str]] = None,
    ) -> None:
        self.size = int(size)
        self.root = root
        self.seed = int(seed)
        self.seq_len = int(seq_len)
        self.max_channels = int(max_channels)
        self.sources = tuple(sources or self.BACKGROUNDS)
        self._windows = self._build_windows()

    def _build_windows(self) -> list[tuple[str, int]]:
        rng = np.random.default_rng(self.seed)
        windows: list[tuple[str, int]] = []
        if "ETTm1" in self.sources:
            ds = ForecastingDataset("ETTm1", split="train", root=self.root, seq_len=self.seq_len, pred_len=1, normalize=True)
            windows.extend(("ETTm1", int(i)) for i in range(len(ds)))
        if "Weather" in self.sources:
            ds = ForecastingDataset("Weather", split="train", root=self.root, seq_len=self.seq_len, pred_len=1, normalize=True)
            windows.extend(("Weather", int(i)) for i in range(len(ds)))
        if "TSB-AD" in self.sources:
            train_files, _ = split_tsb_ad_tuning_files(root=self.root)
            ds = TSBADDataset(
                split="tuning",
                filenames=train_files,
                root=self.root,
                window_size=self.seq_len,
                max_channels=self.max_channels,
                normal_prefix_only=True,
                normalize=True,
            )
            windows.extend(("TSB-AD", int(i)) for i in range(len(ds)))
        if "UCR" in self.sources:
            for name in ("CBF", "SyntheticControl", "ECG5000"):
                ds = UCRDataset(name, split="train", root=self.root, normalize=True)
                windows.extend((f"UCR:{name}", int(i)) for i in range(len(ds)))
        if "UCI_HAR" in self.sources:
            har_root = f"{self.root.rstrip('/')}/UCI_HAR"
            ds = UCIHARDataset(split="train", root=har_root, normalize=True)
            windows.extend(("UCI_HAR", int(i)) for i in range(len(ds)))
        if not windows:
            raise RuntimeError("No real-background windows available for injection.")
        rng.shuffle(windows)
        if len(windows) >= self.size:
            return windows[: self.size]
        repeats = (self.size + len(windows) - 1) // len(windows)
        return (windows * repeats)[: self.size]

    def _load_window(self, source: str, index: int) -> np.ndarray:
        if source == "ETTm1":
            ds = ForecastingDataset("ETTm1", split="train", root=self.root, seq_len=self.seq_len, pred_len=1, normalize=True)
        elif source == "Weather":
            ds = ForecastingDataset("Weather", split="train", root=self.root, seq_len=self.seq_len, pred_len=1, normalize=True)
        elif source == "TSB-AD":
            train_files, _ = split_tsb_ad_tuning_files(root=self.root)
            ds = TSBADDataset(
                split="tuning",
                filenames=train_files,
                root=self.root,
                window_size=self.seq_len,
                max_channels=self.max_channels,
                normal_prefix_only=True,
                normalize=True,
            )
        elif source == "UCI_HAR":
            har_root = f"{self.root.rstrip('/')}/UCI_HAR"
            ds = UCIHARDataset(split="train", root=har_root, normalize=True)
        elif source.startswith("UCR:"):
            name = source.split(":", 1)[1]
            ds = UCRDataset(name, split="train", root=self.root, normalize=True)
        else:
            raise ValueError(f"Unknown background source: {source}")
        sample = ds[index]
        return _fit_window_length(sample.x.numpy(), self.seq_len)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TimeSeriesSample:
        source, window_index = self._windows[index % len(self._windows)]
        rng = np.random.default_rng(self.seed + index * 7919)
        background = self._load_window(source, window_index)
        channels, length = background.shape
        n_events = int(rng.integers(1, 3))
        x = background.copy()
        records: list[list[float]] = []
        for _ in range(n_events):
            kind = str(rng.choice(EVENT_TYPES))
            channel = int(rng.integers(0, channels))
            duration = int(rng.integers(max(8, length // 10), max(9, length // 3)))
            start = int(rng.integers(0, max(1, length - duration)))
            end = start + duration
            amplitude = float(rng.uniform(0.5, 1.8))
            effect = _event_effect(kind, duration, amplitude, rng)
            x[channel, start:end] += effect
            records.append(
                [EVENT_TO_ID[kind], start / length, end / length, channel, amplitude]
            )
        event_tensor = torch.tensor(records, dtype=torch.float32)
        return make_sample(
            x,
            {"source": "injection", "background_source": source},
            f"injection:{source}:{index}",
            events_gt=event_tensor,
        )


def iter_real_forecast_windows(
    root: str,
    *,
    seq_len: int,
    pred_len: int,
    datasets: Sequence[str] = ("ETTm1", "Weather"),
) -> Iterator[TimeSeriesSample]:
    for name in datasets:
        bundle = make_forecasting_datasets(name, root=root, seq_len=seq_len, pred_len=pred_len, normalize=True)
        for split in ("train",):
            dataset = bundle[split]
            for index in range(len(dataset)):
                yield dataset[index]
