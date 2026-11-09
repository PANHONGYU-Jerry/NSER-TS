"""Reproducible controlled structural-causal time-series generator."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


EVENT_TYPES = (
    "stable",
    "trend_up",
    "trend_down",
    "spike",
    "valley",
    "level_shift_up",
    "level_shift_down",
    "periodic",
    "frequency_change",
)
EVENT_TO_ID = {name: i for i, name in enumerate(EVENT_TYPES)}
SPLITS = ("train", "val", "test", "counterfactual", "compositional_ood")
_SPLIT_OFFSET = {name: i * 10_000_019 for i, name in enumerate(SPLITS)}


def _event_effect(kind: str, n: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if kind == "stable":
        return np.zeros(n, dtype=np.float32)
    if kind == "trend_up":
        return amplitude * u
    if kind == "trend_down":
        return -amplitude * u
    if kind == "spike":
        width = max(0.06, float(rng.uniform(0.06, 0.18)))
        return amplitude * np.exp(-0.5 * ((u - 0.5) / width) ** 2)
    if kind == "valley":
        width = max(0.06, float(rng.uniform(0.06, 0.18)))
        return -amplitude * np.exp(-0.5 * ((u - 0.5) / width) ** 2)
    if kind == "level_shift_up":
        return amplitude * (u >= 0.25)
    if kind == "level_shift_down":
        return -amplitude * (u >= 0.25)
    if kind == "periodic":
        cycles = int(rng.integers(2, 6))
        return amplitude * np.sin(2 * np.pi * cycles * u)
    if kind == "frequency_change":
        phase = np.where(u < 0.5, 2 * np.pi * 2 * u, 2 * np.pi * (2 * 0.5 + 7 * (u - 0.5)))
        return amplitude * np.sin(phase)
    raise ValueError(f"Unknown event type: {kind}")


class ControlledSCMDataset(Dataset[TimeSeriesSample]):
    """Generate samples by index, with no materialized sample cache.

    ``events_gt`` columns are ``[type_id, start, end, channel, amplitude]``;
    start/end are normalized boundaries in ``[0, 1]``. ``relations_gt`` rows
    are directed SCM edges ``[source_channel, target_channel, weight]``.
    """

    def __init__(
        self,
        size: int = 10_000,
        split: str = "train",
        seed: int = 42,
        length_range: tuple[int, int] = (128, 512),
        channel_range: tuple[int, int] = (1, 6),
        event_range: tuple[int, int] = (1, 4),
        event_types: Optional[Sequence[str]] = None,
        return_counterfactual: bool = False,
        noise_std: float = 0.08,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.size = int(size)
        self.split = split
        self.seed = int(seed)
        self.length_range = length_range
        self.channel_range = channel_range
        self.event_range = event_range
        self.event_types = tuple(event_types or EVENT_TYPES)
        unknown = set(self.event_types) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"Unknown events: {sorted(unknown)}")
        self.return_counterfactual = return_counterfactual or split == "counterfactual"
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return self.size

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + _SPLIT_OFFSET[self.split] + int(index))

    def __getitem__(self, index: int) -> TimeSeriesSample:
        if not 0 <= index < self.size:
            raise IndexError(index)
        rng = self._rng(index)
        t_len = int(rng.integers(self.length_range[0], self.length_range[1] + 1))
        n_channels = int(rng.integers(self.channel_range[0], self.channel_range[1] + 1))
        low_events = max(self.event_range[0], 2 if self.split == "compositional_ood" else 1)
        n_events = int(rng.integers(low_events, self.event_range[1] + 1))

        time = np.linspace(0, 1, t_len, dtype=np.float32)
        adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)
        for target in range(1, n_channels):
            if rng.random() < 0.7:
                source = int(rng.integers(0, target))
                adjacency[source, target] = float(rng.uniform(-0.65, 0.65))
        baseline = np.empty((n_channels, t_len), dtype=np.float32)
        for channel in range(n_channels):
            baseline[channel] = (
                rng.normal(0, self.noise_std, t_len)
                + rng.uniform(-0.25, 0.25) * time
                + rng.uniform(-0.3, 0.3)
            )
            for source in range(channel):
                baseline[channel] += adjacency[source, channel] * baseline[source]

        x = baseline.copy()
        records: list[list[float]] = []
        kinds: list[str] = []
        direct_effects: list[np.ndarray] = []
        if self.split == "compositional_ood":
            # Deliberately combine distinct shape families in one sample.
            families = [
                ("trend_up", "spike"),
                ("trend_down", "level_shift_up"),
                ("periodic", "frequency_change"),
                ("valley", "level_shift_down"),
            ]
            forced = families[int(rng.integers(0, len(families)))]
            kinds.extend(forced[:n_events])
        while len(kinds) < n_events:
            kinds.append(str(rng.choice(self.event_types)))

        for kind in kinds:
            channel = int(rng.integers(0, n_channels))
            duration = int(rng.integers(max(12, t_len // 12), max(13, t_len // 3)))
            start = int(rng.integers(0, t_len - duration + 1))
            end = start + duration
            amplitude = float(rng.uniform(0.6, 2.0))
            direct_effect = _event_effect(kind, duration, amplitude, rng)
            x[channel, start:end] += direct_effect
            # Propagate interventions through direct SCM children.
            for target in range(channel + 1, n_channels):
                if adjacency[channel, target] != 0:
                    x[target, start:end] += adjacency[channel, target] * direct_effect
            records.append([EVENT_TO_ID[kind], start / t_len, end / t_len, channel, amplitude])
            direct_effects.append(direct_effect)

        event_tensor = torch.tensor(records, dtype=torch.float32)
        multi_hot = torch.zeros(len(EVENT_TYPES), dtype=torch.float32)
        multi_hot[event_tensor[:, 0].long()] = 1.0
        label: dict[str, object] = {"event_types": multi_hot, "split": self.split}
        if self.return_counterfactual:
            # Intervention do(event_0 := absent), with identical exogenous noise.
            cf = x.copy()
            first = records[0]
            kind, start, end, channel = kinds[0], int(first[1] * t_len), int(first[2] * t_len), int(first[3])
            direct_effect = direct_effects[0]
            cf[channel, start:end] -= direct_effect
            for target in range(channel + 1, n_channels):
                cf[target, start:end] -= adjacency[channel, target] * direct_effect
            label["counterfactual"] = torch.from_numpy(cf)
            label["intervened_event"] = EVENT_TO_ID[kind]

        edges = np.argwhere(adjacency != 0)
        relations = torch.tensor(
            [[int(s), int(t), float(adjacency[s, t])] for s, t in edges],
            dtype=torch.float32,
        ).reshape(-1, 3)
        return make_sample(
            x,
            label,
            f"controlled_scm:{self.split}:{index}",
            events_gt=event_tensor,
            relations_gt=relations,
        )


def make_controlled_scm(**kwargs: object) -> ControlledSCMDataset:
    return ControlledSCMDataset(**kwargs)
