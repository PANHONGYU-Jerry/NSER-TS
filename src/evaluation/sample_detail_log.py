"""Human-readable sample dumps for training / evaluation runs."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.event_postprocess import (
    BACKGROUND_TYPES,
    EventPostprocessConfig,
    postprocess_event_nodes,
)
from src.event_schema import ATTRIBUTE_NAMES, PRIMITIVES


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _batch_sample_ids(batch: Mapping[str, Any], batch_size: int) -> List[str]:
    sample_ids = batch.get("sample_id")
    if sample_ids is None:
        return [f"sample:{index}" for index in range(batch_size)]
    if isinstance(sample_ids, str):
        return [sample_ids]
    return [str(item) for item in sample_ids]


def _primary_channel(
    x: torch.Tensor, channel_mask: torch.Tensor, batch_index: int
) -> int:
    channels = int(channel_mask[batch_index].sum().item())
    if channels <= 0:
        return 0
    valid = x[batch_index, :channels].detach().cpu()
    variances = valid.var(dim=-1)
    return int(variances.argmax().item())


def _subsample_series(values: Sequence[float], max_points: int) -> List[float]:
    if len(values) <= max_points:
        return [float(value) for value in values]
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return [float(values[index]) for index in indices]


def _time_axis(
    batch: Mapping[str, Any],
    batch_index: int,
    length: int,
    max_points: int,
) -> List[float]:
    """Always use a monotonic normalized axis for visualization."""
    del batch, batch_index
    if length <= 1:
        return [0.0]
    normalized = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return _subsample_series(normalized.tolist(), max_points)


def _channels_of_interest(
    event_nodes: Sequence[Mapping[str, Any]],
    events_gt: Optional[Sequence[Mapping[str, Any]]],
    channel_mask: torch.Tensor,
    batch_index: int,
    *,
    max_channels: int,
) -> List[int]:
    valid = int(channel_mask[batch_index].sum().item())
    channels: set[int] = set()
    for event in event_nodes:
        channels.add(int(event.get("channel", 0)))
    if events_gt:
        for event in events_gt:
            channels.add(int(event.get("channel", 0)))
    ordered = sorted(channel for channel in channels if 0 <= channel < valid)
    return ordered[:max_channels]


def _extract_time_series_by_channel(
    x: torch.Tensor,
    batch_index: int,
    valid_length: int,
    channels: Sequence[int],
    time_axis: Sequence[float],
    max_points: int,
) -> Dict[str, Dict[str, List[float]]]:
    series: Dict[str, Dict[str, List[float]]] = {}
    for channel in channels:
        flux = x[batch_index, channel, :valid_length].detach().cpu().tolist()
        series[str(channel)] = {
            "time": list(time_axis),
            "flux": _subsample_series(flux, max_points),
        }
    return series


def _derived_magnitude_series(
    x: torch.Tensor,
    batch_index: int,
    valid_length: int,
    time_axis: Sequence[float],
    max_points: int,
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Body-acc magnitude (first 3 channels) as a derived plot channel.

    UCI HAR oracle labels are computed on body-acc magnitude, not on any single
    raw channel; exposing the magnitude lets the GT interval overlay line up
    with the signal the labels were actually derived from.
    """
    if x.ndim < 2 or x.shape[1] < 3:
        return None
    body = x[batch_index, :3, :valid_length]
    mag = torch.sqrt((body * body).sum(dim=0)).detach().cpu().tolist()
    return {
        "mag": {
            "time": list(time_axis),
            "flux": _subsample_series(mag, max_points),
        }
    }


def _detail_postprocess_config() -> EventPostprocessConfig:
    import os

    return EventPostprocessConfig(
        existence_threshold=float(os.environ.get("EVENT_EXISTENCE_THRESHOLD", "0.5")),
    )


def _extract_events(
    events: Mapping[str, torch.Tensor],
    batch_index: int,
    valid_length: int,
    existence_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], bool]:
    existence = events["existence"][batch_index].detach().cpu()
    selected = torch.nonzero(existence >= existence_threshold, as_tuple=False).flatten()
    has_valid_events = selected.numel() > 0

    result: List[Dict[str, Any]] = []
    for slot in selected.tolist():
        probabilities = events["type"][batch_index, slot].detach().cpu()
        attrs = events["attributes"][batch_index, slot].detach().cpu()
        attr_mask = events["attribute_mask"][batch_index, slot].detach().cpu()
        start = float(events["start"][batch_index, slot].detach())
        end = float(events["end"][batch_index, slot].detach())
        start_index = int(round(start * max(valid_length - 1, 0)))
        end_index = int(round(end * max(valid_length - 1, 0)))
        result.append(
            {
                "slot": int(slot),
                "existence": float(existence[slot]),
                "type": PRIMITIVES[int(probabilities.argmax()) % len(PRIMITIVES)],
                "type_probs": {
                    PRIMITIVES[index % len(PRIMITIVES)]: float(probabilities[index])
                    for index in range(min(len(PRIMITIVES), probabilities.numel()))
                },
                "start": start,
                "end": end,
                "start_index": start_index,
                "end_index": end_index,
                "channel": int(events["channel_scores"][batch_index, slot].detach().argmax()),
                "attributes": {
                    name: float(attrs[index])
                    for index, name in enumerate(ATTRIBUTE_NAMES)
                    if index < attrs.numel()
                    and index < attr_mask.numel()
                    and float(attr_mask[index]) >= 0.5
                },
                "confidence": float(events["confidence"][batch_index, slot].detach()),
                "layer": (
                    "background"
                    if PRIMITIVES[int(probabilities.argmax()) % len(PRIMITIVES)]
                    in BACKGROUND_TYPES
                    else "foreground"
                ),
            }
        )
    return result, has_valid_events


def _extract_relation_graph(
    graph: Mapping[str, Any],
    batch_index: int,
    event_nodes: Sequence[Mapping[str, Any]],
    *,
    deterministic_threshold: float = 0.5,
    learned_threshold: float = 0.75,
) -> Dict[str, Any]:
    if not event_nodes:
        return {"nodes": [], "edges": [], "ascii": ""}

    selected = [int(node["slot"]) for node in event_nodes]
    slot_to_local = {slot: local for local, slot in enumerate(selected)}
    scores = graph["relation_scores"][batch_index].detach().cpu()
    names = graph["relation_names"]
    edges: List[Dict[str, Any]] = []
    ascii_lines: List[str] = []
    selected_set = set(selected)
    for source in selected:
        for target in selected:
            if source == target:
                continue
            for relation_index, name in enumerate(names):
                score = float(scores[source, target, relation_index])
                threshold = (
                    deterministic_threshold
                    if relation_index < 7
                    else learned_threshold
                )
                if score >= threshold and source in selected_set and target in selected_set:
                    edge = {
                        "source_slot": source,
                        "target_slot": target,
                        "source": slot_to_local[source],
                        "target": slot_to_local[target],
                        "relation": name,
                        "score": score,
                    }
                    edges.append(edge)
                    source_type = next(
                        node["type"] for node in event_nodes if node["slot"] == source
                    )
                    target_type = next(
                        node["type"] for node in event_nodes if node["slot"] == target
                    )
                    ascii_lines.append(
                        f"E{source}({source_type}) --{name}({score:.2f})--> "
                        f"E{target}({target_type})"
                    )
    return {
        "nodes": event_nodes,
        "edges": edges,
        "ascii": "\n".join(ascii_lines),
    }


def _extract_events_gt(
    events_gt: Any,
    batch_index: int,
    valid_length: int,
) -> Optional[List[Dict[str, Any]]]:
    if events_gt is None:
        return None
    # Morphology supervision targets arrive as a collated dict of per-key
    # tensors/lists; controlled-SCM ground truth is row tensors [n_events, 5].
    if isinstance(events_gt, Mapping):
        types = _value_at_label(events_gt.get("types"), batch_index)
        if types is None or not torch.is_tensor(types) or types.numel() == 0:
            return None
        boundaries = _value_at_label(events_gt.get("boundaries"), batch_index)
        channels = _value_at_label(events_gt.get("channels"), batch_index)
        event_channels = _value_at_label(events_gt.get("event_channels"), batch_index)
        result: List[Dict[str, Any]] = []
        for index in range(types.shape[0]):
            start = float(boundaries[index, 0]) if boundaries is not None else 0.0
            end = float(boundaries[index, 1]) if boundaries is not None else start
            if event_channels is not None and event_channels.numel() > index:
                channel = int(event_channels[index])
            elif channels is not None and channels.ndim >= 2:
                channel = int(channels[index].argmax())
            elif channels is not None:
                channel = int(channels[index])
            else:
                channel = 0
            result.append(
                {
                    "index": index,
                    "type": PRIMITIVES[int(types[index]) % len(PRIMITIVES)],
                    "start": start,
                    "end": end,
                    "start_index": int(round(start * max(valid_length - 1, 0))),
                    "end_index": int(round(end * max(valid_length - 1, 0))),
                    "channel": channel,
                    "amplitude": None,
                }
            )
        return result
    rows = events_gt[batch_index] if isinstance(events_gt, list) else events_gt[batch_index]
    if rows is None or not torch.is_tensor(rows) or rows.numel() == 0:
        return None
    rows = rows.detach().cpu()
    result: List[Dict[str, Any]] = []
    for index in range(rows.shape[0]):
        start = float(rows[index, 1])
        end = float(rows[index, 2])
        result.append(
            {
                "index": index,
                "type": PRIMITIVES[int(rows[index, 0]) % len(PRIMITIVES)],
                "start": start,
                "end": end,
                "start_index": int(round(start * max(valid_length - 1, 0))),
                "end_index": int(round(end * max(valid_length - 1, 0))),
                "channel": int(rows[index, 3]) if rows.shape[1] > 3 else 0,
                "amplitude": float(rows[index, 4]) if rows.shape[1] > 4 else None,
            }
        )
    return result


def _value_at_label(value: Any, batch_index: int) -> Any:
    if torch.is_tensor(value):
        return value[batch_index].detach().cpu()
    if isinstance(value, list):
        return value[batch_index]
    return value


def build_sample_detail_record(
    batch: Mapping[str, Any],
    output: Mapping[str, Any],
    batch_index: int,
    *,
    task: str,
    phase: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    max_points: int = 128,
    max_channels: int = 6,
) -> Dict[str, Any]:
    x = batch["x"]
    time_mask = batch["time_mask"]
    channel_mask = batch["channel_mask"]
    valid_length = int(time_mask[batch_index].sum().item())
    time_axis = _time_axis(batch, batch_index, valid_length, max_points)

    events = output["events"]
    postprocess_config = _detail_postprocess_config()
    raw_event_nodes, has_valid_events = _extract_events(
        events,
        batch_index,
        valid_length,
        existence_threshold=postprocess_config.existence_threshold,
    )
    event_nodes, rejected_events = postprocess_event_nodes(
        raw_event_nodes,
        x[batch_index].detach().cpu().numpy(),
        valid_length,
        config=postprocess_config,
    )
    events_gt = _extract_events_gt(batch.get("events_gt"), batch_index, valid_length)
    channels = _channels_of_interest(
        event_nodes,
        events_gt,
        channel_mask,
        batch_index,
        max_channels=max_channels,
    )
    if not channels:
        channel = _primary_channel(x, channel_mask, batch_index)
        channels = [channel]
    time_series_by_channel = _extract_time_series_by_channel(
        x,
        batch_index,
        valid_length,
        channels,
        time_axis,
        max_points,
    )
    gt_series_key: Optional[str] = None
    sample_id = _batch_sample_ids(batch, x.shape[0])[batch_index]
    if str(sample_id).startswith("uci_har:"):
        mag_series = _derived_magnitude_series(
            x, batch_index, valid_length, time_axis, max_points
        )
        if mag_series is not None:
            time_series_by_channel.update(mag_series)
            channels = list(channels) + ["mag"]
            gt_series_key = "mag"
    primary_channel = channels[0]
    graph = output.get("graph", {})
    relation_graph = _extract_relation_graph(graph, batch_index, event_nodes)

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "task": task,
        "epoch": epoch,
        "step": step,
        "sample_id": _batch_sample_ids(batch, x.shape[0])[batch_index],
        "valid_length": valid_length,
        "primary_channel": primary_channel,
        "channels_plotted": channels,
        "gt_series_key": gt_series_key,
        "time_series": time_series_by_channel[str(primary_channel)],
        "time_series_by_channel": time_series_by_channel,
        "events": event_nodes,
        "events_raw": raw_event_nodes,
        "events_rejected": rejected_events,
        "events_valid": has_valid_events and bool(event_nodes),
        "relation_graph": relation_graph,
        "events_gt": events_gt,
    }
    if "task_label" in batch:
        label = batch["task_label"]
        if isinstance(label, list):
            record["task_label"] = _json_safe(label[batch_index])
        elif torch.is_tensor(label):
            record["task_label"] = _json_safe(label[batch_index].detach().cpu())
        elif isinstance(label, Mapping):
            record["task_label"] = _json_safe(
                {key: _value_at_label(value, batch_index) for key, value in label.items()}
            )
        else:
            record["task_label"] = _json_safe(label)
    return _json_safe(record)


class SampleDetailLogger:
    """Append structured sample dumps to ``detail.log``."""

    def __init__(
        self,
        path: str | Path,
        *,
        samples_per_epoch: int = 2,
        max_points: int = 128,
        max_channels: int = 6,
        min_epoch: int = 1,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.samples_per_epoch = max(0, samples_per_epoch)
        self.max_points = max(8, max_points)
        self.max_channels = max(1, max_channels)
        self.min_epoch = max(1, min_epoch)
        self.rng = random.Random(seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._opened = False

    @classmethod
    def from_env(cls, seed: int = 42) -> Optional["SampleDetailLogger"]:
        import os

        path = os.environ.get("DETAIL_LOG", "").strip()
        if not path:
            return None
        samples = int(os.environ.get("DETAIL_SAMPLES", "2"))
        max_points = int(os.environ.get("DETAIL_MAX_POINTS", "128"))
        max_channels = int(os.environ.get("DETAIL_MAX_CHANNELS", "6"))
        min_epoch = int(os.environ.get("DETAIL_MIN_EPOCH", "1"))
        return cls(
            path,
            samples_per_epoch=samples,
            max_points=max_points,
            max_channels=max_channels,
            min_epoch=min_epoch,
            seed=seed,
        )

    def _ensure_header(self) -> None:
        if self._opened:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"# sample detail log | created {datetime.now(timezone.utc).isoformat()}\n"
            )
        self._opened = True

    def write_record(self, record: Mapping[str, Any]) -> None:
        self._ensure_header()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
            handle.write("\n---\n")

    def log_batch(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
        *,
        task: str,
        phase: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        sample_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if self.samples_per_epoch <= 0:
            return
        batch_size = batch["x"].shape[0]
        if sample_indices is None:
            count = min(self.samples_per_epoch, batch_size)
            sample_indices = self.rng.sample(range(batch_size), count)
        for batch_index in sample_indices:
            record = build_sample_detail_record(
                batch,
                output,
                batch_index,
                task=task,
                phase=phase,
                epoch=epoch,
                step=step,
                max_points=self.max_points,
                max_channels=self.max_channels,
            )
            self.write_record(record)

    def planned_train_steps(self, train_steps: int, epochs: int) -> Dict[int, set[int]]:
        if self.samples_per_epoch <= 0 or train_steps <= 0 or epochs <= 0:
            return {}
        planned: Dict[int, set[int]] = {}
        start_epoch = min(max(self.min_epoch, 1), epochs)
        for epoch in range(start_epoch, epochs + 1):
            count = min(self.samples_per_epoch, train_steps)
            planned[epoch] = set(
                self.rng.sample(range(1, train_steps + 1), count)
            )
        return planned
