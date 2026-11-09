"""P1 event post-processing: degenerate rejection, grouped NMS, width priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from src.training.matcher import interval_iou_1d
from src.event_schema import PRIMITIVES

BACKGROUND_TYPES: Set[str] = {"periodic", "frequency_change"}

FOREGROUND_NMS_GROUPS: Tuple[Set[str], ...] = (
    {"spike", "valley"},
    {"level_shift_up", "level_shift_down"},
    {"trend_up", "trend_down", "stable"},
)

TYPE_MAX_REL_WIDTH: Dict[str, float] = {
    "spike": 0.20,
    "valley": 0.20,
    "level_shift_up": 0.30,
    "level_shift_down": 0.30,
    "trend_up": 1.0,
    "trend_down": 1.0,
    "stable": 1.0,
    "periodic": 1.0,
    "frequency_change": 1.0,
}

TYPE_MIN_REL_WIDTH: Dict[str, float] = {
    "spike": 0.005,
    "valley": 0.005,
    "level_shift_up": 0.01,
    "level_shift_down": 0.01,
    "trend_up": 0.03,
    "trend_down": 0.03,
    "stable": 0.03,
    "periodic": 0.12,
    "frequency_change": 0.15,
}

MIN_VALID_POINTS = 3
PERIODIC_MIN_POINTS = 8
DEGENERATE_WIDTH = 1e-4
FOREGROUND_SPLIT_WIDTH = 0.35
NMS_IOU_THRESHOLD = 0.5


def _type_name(type_probs: torch.Tensor) -> str:
    index = int(type_probs.argmax().item())
    return PRIMITIVES[index % len(PRIMITIVES)]


def _same_nms_group(type_a: str, type_b: str) -> bool:
    if type_a == type_b:
        return True
    for group in FOREGROUND_NMS_GROUPS:
        if type_a in group and type_b in group:
            return True
    return False


def _interval_width(start: float, end: float) -> float:
    return max(float(end) - float(start), 0.0)


@dataclass(frozen=True)
class EventPostprocessConfig:
    existence_threshold: float = 0.5
    nms_iou_threshold: float = NMS_IOU_THRESHOLD
    min_valid_points: int = MIN_VALID_POINTS
    periodic_min_points: int = PERIODIC_MIN_POINTS
    degenerate_width: float = DEGENERATE_WIDTH
    split_foreground_width: float = FOREGROUND_SPLIT_WIDTH


def _valid_length_mask(
    start: torch.Tensor,
    end: torch.Tensor,
    valid_length: int,
    *,
    min_points: int,
    min_rel_width: float,
) -> torch.Tensor:
    width = (end - start).clamp_min(0.0)
    min_width = min_rel_width
    if valid_length > 1:
        min_width = max(min_width, float(min_points - 1) / float(valid_length - 1))
    return (width >= min_width) & (width >= DEGENERATE_WIDTH)


def degenerate_mask(
    events: Mapping[str, torch.Tensor],
    batch_index: int,
    valid_length: int,
    *,
    config: EventPostprocessConfig,
) -> torch.Tensor:
    """Return bool mask [K] — True means slot survives degenerate checks."""
    start = events["start"][batch_index]
    end = events["end"][batch_index]
    type_probs = events["type"][batch_index]
    k = start.shape[0]
    keep = torch.ones(k, dtype=torch.bool, device=start.device)
    for slot in range(k):
        event_type = _type_name(type_probs[slot])
        min_rel = TYPE_MIN_REL_WIDTH.get(event_type, 0.01)
        min_points = (
            config.periodic_min_points
            if event_type in BACKGROUND_TYPES
            else config.min_valid_points
        )
        if not bool(
            _valid_length_mask(
                start[slot : slot + 1],
                end[slot : slot + 1],
                valid_length,
                min_points=min_points,
                min_rel_width=min_rel,
            ).item()
        ):
            keep[slot] = False
    return keep


def grouped_foreground_nms(
    slots: Sequence[int],
    events: Mapping[str, torch.Tensor],
    batch_index: int,
    *,
    config: EventPostprocessConfig,
) -> List[int]:
    """Suppress overlapping foreground slots within the same NMS group."""
    if len(slots) <= 1:
        return list(slots)

    existence = events["existence"][batch_index]
    type_probs = events["type"][batch_index]
    boundaries = torch.stack(
        [events["start"][batch_index], events["end"][batch_index]], dim=-1
    )
    channels = events["channel_scores"][batch_index].argmax(dim=-1)

    ordered = sorted(
        slots,
        key=lambda slot: (
            float(existence[slot].item()),
            float(events["confidence"][batch_index, slot].item()),
        ),
        reverse=True,
    )
    kept: List[int] = []
    for slot in ordered:
        event_type = _type_name(type_probs[slot])
        channel = int(channels[slot].item())
        suppress = False
        for other in kept:
            if int(channels[other].item()) != channel:
                continue
            other_type = _type_name(type_probs[other])
            if not _same_nms_group(event_type, other_type):
                continue
            iou = interval_iou_1d(
                boundaries[slot : slot + 1], boundaries[other : other + 1]
            ).item()
            if iou >= config.nms_iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(slot)
    return kept


def _split_points(signal: np.ndarray, start_index: int, end_index: int) -> List[int]:
    segment = signal[start_index : end_index + 1]
    if segment.size < 6:
        return []
    diff = np.abs(np.diff(segment.astype(np.float64)))
    if diff.size == 0:
        return []
    threshold = float(np.percentile(diff, 75))
    if threshold <= 0:
        return []
    peaks = np.where(diff >= threshold)[0] + 1
    if peaks.size == 0:
        mid = segment.size // 2
        return [start_index + mid]
    return [start_index + int(point) for point in peaks.tolist()]


def split_wide_foreground_event(
    event: Mapping[str, Any],
    signal: np.ndarray,
    valid_length: int,
    *,
    split_width: float = FOREGROUND_SPLIT_WIDTH,
) -> List[Dict[str, Any]]:
    """Split an overly wide foreground interval at local change points."""
    event_type = str(event.get("type", ""))
    if event_type in BACKGROUND_TYPES:
        return [dict(event)]
    start = float(event.get("start", 0.0))
    end = float(event.get("end", 0.0))
    if _interval_width(start, end) <= split_width:
        return [dict(event)]

    start_index = int(round(start * max(valid_length - 1, 0)))
    end_index = int(round(end * max(valid_length - 1, 0)))
    if end_index <= start_index + 2:
        return [dict(event)]

    channel = int(event.get("channel", 0))
    if signal.ndim == 2:
        trace = signal[channel]
    else:
        trace = signal
    points = _split_points(trace, start_index, end_index)
    if not points:
        return [dict(event)]

    boundaries = [start_index] + points + [end_index]
    pieces: List[Dict[str, Any]] = []
    denom = max(valid_length - 1, 1)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        piece = dict(event)
        piece["start"] = float(left) / denom
        piece["end"] = float(right) / denom
        piece["start_index"] = left
        piece["end_index"] = right
        piece["split_from"] = event.get("slot", event.get("index"))
        pieces.append(piece)
    return pieces or [dict(event)]


def postprocess_event_nodes(
    event_nodes: Sequence[Mapping[str, Any]],
    raw_x: np.ndarray,
    valid_length: int,
    *,
    config: Optional[EventPostprocessConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter and split list-form events for detail logs."""
    config = config or EventPostprocessConfig()
    if raw_x.ndim == 3:
        signal = raw_x[0]
    elif raw_x.ndim == 2:
        signal = raw_x
    else:
        signal = raw_x.reshape(1, -1)

    foreground: List[Dict[str, Any]] = []
    background: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for event in event_nodes:
        existence = float(event.get("existence", 1.0))
        if existence < config.existence_threshold:
            rejected.append({**dict(event), "reject_reason": "low_existence"})
            continue
        event_type = str(event.get("type", ""))
        width = _interval_width(event.get("start", 0.0), event.get("end", 0.0))
        min_rel = TYPE_MIN_REL_WIDTH.get(event_type, 0.01)
        if width < max(min_rel, config.degenerate_width):
            rejected.append({**dict(event), "reject_reason": "degenerate_width"})
            continue
        start_index = int(event.get("start_index", 0))
        end_index = int(event.get("end_index", 0))
        min_points = (
            config.periodic_min_points
            if event_type in BACKGROUND_TYPES
            else config.min_valid_points
        )
        if end_index - start_index + 1 < min_points:
            rejected.append({**dict(event), "reject_reason": "too_few_points"})
            continue

        pieces = (
            [dict(event)]
            if event_type in BACKGROUND_TYPES
            else split_wide_foreground_event(
                event,
                signal,
                valid_length,
                split_width=config.split_foreground_width,
            )
        )
        for piece in pieces:
            if event_type in BACKGROUND_TYPES:
                background.append(piece)
            else:
                foreground.append(piece)

    # Grouped NMS on foreground list events
    ordered = sorted(
        foreground,
        key=lambda item: (
            float(item.get("existence", 0.0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    kept_foreground: List[Dict[str, Any]] = []
    for event in ordered:
        event_type = str(event.get("type", ""))
        channel = int(event.get("channel", 0))
        suppress = False
        for other in kept_foreground:
            if int(other.get("channel", 0)) != channel:
                continue
            if not _same_nms_group(event_type, str(other.get("type", ""))):
                continue
            left = max(float(event.get("start", 0.0)), float(other.get("start", 0.0)))
            right = min(float(event.get("end", 0.0)), float(other.get("end", 0.0)))
            inter = max(0.0, right - left)
            union = _interval_width(event.get("start", 0.0), event.get("end", 0.0)) + _interval_width(
                other.get("start", 0.0), other.get("end", 0.0)
            ) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= config.nms_iou_threshold:
                suppress = True
                break
        if suppress:
            rejected.append({**event, "reject_reason": "nms_suppressed"})
        else:
            kept_foreground.append(event)

    merged = background + kept_foreground
    merged.sort(key=lambda item: (int(item.get("channel", 0)), float(item.get("start", 0.0))))
    return merged, rejected


def postprocess_events_tensor(
    events: Mapping[str, torch.Tensor],
    *,
    time_mask: Optional[torch.Tensor] = None,
    config: Optional[EventPostprocessConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Zero existence for degenerate / NMS-suppressed slots (inference-safe)."""
    config = config or EventPostprocessConfig()
    result = {key: value for key, value in events.items()}
    existence = result["existence"].clone()
    batch_size, num_slots = existence.shape
    for batch_index in range(batch_size):
        valid_length = (
            int(time_mask[batch_index].sum().item())
            if time_mask is not None
            else existence.shape[-1]
        )
        active = torch.nonzero(
            existence[batch_index] >= config.existence_threshold, as_tuple=False
        ).flatten()
        if active.numel() == 0:
            continue
        keep = degenerate_mask(result, batch_index, valid_length, config=config)
        active = active[keep[active]]
        if active.numel() == 0:
            existence[batch_index] = 0.0
            continue

        type_probs = result["type"][batch_index]
        background_slots = [
            int(slot)
            for slot in active.tolist()
            if _type_name(type_probs[slot]) in BACKGROUND_TYPES
        ]
        foreground_slots = [
            int(slot)
            for slot in active.tolist()
            if _type_name(type_probs[slot]) not in BACKGROUND_TYPES
        ]
        kept = set(background_slots)
        kept.update(grouped_foreground_nms(foreground_slots, result, batch_index, config=config))
        for slot in range(num_slots):
            if slot not in kept:
                existence[batch_index, slot] = 0.0
    result["existence"] = existence
    return result


def event_width_prior_loss(events: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Soft penalty when predicted intervals violate type-specific width priors."""
    existence = events["existence"]
    type_probs = events["type"]
    start = events["start"]
    end = events["end"]
    width = (end - start).clamp_min(0.0)
    penalties = []
    for type_index, primitive in enumerate(PRIMITIVES):
        if type_index >= type_probs.shape[-1]:
            break
        weight = type_probs[..., type_index]
        max_width = TYPE_MAX_REL_WIDTH.get(primitive, 1.0)
        min_width = TYPE_MIN_REL_WIDTH.get(primitive, 0.01)
        too_wide = torch.relu(width - max_width)
        too_narrow = torch.relu(min_width - width)
        slot_penalty = (too_wide + too_narrow) * weight * existence
        penalties.append(slot_penalty)
    if not penalties:
        return existence.sum() * 0.0
    return torch.stack(penalties, dim=0).sum(dim=0).mean()
