"""Common sample protocol and variable-size collation for time-series tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass
class TimeSeriesSample:
    """A model-independent time-series sample.

    ``x`` is always channel-first ``[C, T]``. Event boundaries stored in
    ``events_gt`` are normalized to the closed interval ``[0, 1]``.
    """

    x: torch.Tensor
    time_mask: torch.Tensor
    channel_mask: torch.Tensor
    task_label: Any
    sample_id: str
    events_gt: Optional[Any] = None
    relations_gt: Optional[Any] = None
    time_features: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.x = torch.as_tensor(self.x, dtype=torch.float32)
        self.time_mask = torch.as_tensor(self.time_mask, dtype=torch.bool)
        self.channel_mask = torch.as_tensor(self.channel_mask, dtype=torch.bool)
        if self.x.ndim != 2:
            raise ValueError(f"x must have shape [C,T], got {tuple(self.x.shape)}")
        if tuple(self.time_mask.shape) != (self.x.shape[1],):
            raise ValueError("time_mask must have shape [T]")
        if tuple(self.channel_mask.shape) != (self.x.shape[0],):
            raise ValueError("channel_mask must have shape [C]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_sample(
    x: Any,
    task_label: Any,
    sample_id: str,
    *,
    events_gt: Any = None,
    relations_gt: Any = None,
    time_features: Any = None,
) -> TimeSeriesSample:
    """Build a fully-observed sample and its masks."""
    tensor = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return TimeSeriesSample(
        x=tensor,
        time_mask=torch.ones(tensor.shape[1], dtype=torch.bool),
        channel_mask=torch.ones(tensor.shape[0], dtype=torch.bool),
        task_label=task_label,
        events_gt=events_gt,
        relations_gt=relations_gt,
        time_features=torch.as_tensor(time_features, dtype=torch.float32)
        if time_features is not None
        else None,
        sample_id=str(sample_id),
    )


def _collate_values(values: Sequence[Any]) -> Any:
    """Recursively collate labels/ground truth without assuming one task."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        return list(values)
    first = values[0]
    if isinstance(first, Mapping) and all(isinstance(v, Mapping) for v in values):
        keys = set(first)
        if all(set(v) == keys for v in values):
            return {key: _collate_values([v[key] for v in values]) for key in keys}
        return list(values)
    if isinstance(first, (torch.Tensor, np.ndarray, int, float, bool, np.number)):
        tensors = [torch.as_tensor(v) for v in values]
        if all(t.shape == tensors[0].shape for t in tensors):
            return torch.stack(tensors)
        return tensors
    return list(values)


def padding_collate(
    batch: Sequence[TimeSeriesSample | Mapping[str, Any]],
    *,
    pad_value: float = 0.0,
) -> dict[str, Any]:
    """Pad both channel and time axes to the largest sample in a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    items = [item.as_dict() if isinstance(item, TimeSeriesSample) else dict(item) for item in batch]
    xs = [torch.as_tensor(item["x"], dtype=torch.float32) for item in items]
    max_c = max(x.shape[0] for x in xs)
    max_t = max(x.shape[1] for x in xs)
    out_x = torch.full((len(xs), max_c, max_t), float(pad_value), dtype=torch.float32)
    time_mask = torch.zeros((len(xs), max_t), dtype=torch.bool)
    channel_mask = torch.zeros((len(xs), max_c), dtype=torch.bool)
    for i, (item, x) in enumerate(zip(items, xs)):
        c, t = x.shape
        out_x[i, :c, :t] = x
        tm = torch.as_tensor(item.get("time_mask", torch.ones(t)), dtype=torch.bool)
        cm = torch.as_tensor(item.get("channel_mask", torch.ones(c)), dtype=torch.bool)
        time_mask[i, :t] = tm
        channel_mask[i, :c] = cm
    time_features = [item.get("time_features") for item in items]
    padded_time_features = None
    if any(value is not None for value in time_features):
        feature_dim = max(
            torch.as_tensor(value).shape[-1]
            for value in time_features
            if value is not None
        )
        padded_time_features = torch.zeros(
            len(items), max_t, feature_dim, dtype=torch.float32
        )
        for index, value in enumerate(time_features):
            if value is None:
                continue
            tensor = torch.as_tensor(value, dtype=torch.float32)
            padded_time_features[index, : tensor.shape[0], : tensor.shape[1]] = tensor
    return {
        "x": out_x,
        "time_mask": time_mask,
        "channel_mask": channel_mask,
        "task_label": _collate_values([item.get("task_label") for item in items]),
        "events_gt": _collate_values([item.get("events_gt") for item in items]),
        "relations_gt": _collate_values([item.get("relations_gt") for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "time_features": padded_time_features,
    }


collate_fn = padding_collate
