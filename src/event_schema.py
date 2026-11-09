"""Event type / attribute schema and raw-signal measurements."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


PRIMITIVES = (
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

ATTRIBUTE_NAMES = (
    "amplitude",
    "slope",
    "duration",
    "delta_mean",
    "prominence",
    "frequency",
    "phase",
)

EPS = 1e-9

def to_numpy(value: Any) -> np.ndarray:
    """Convert a numpy array, torch tensor, or sequence to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def signal_1d(raw_x: Any, channel: Optional[int] = None) -> np.ndarray:
    """Extract one time channel, treating the longest non-singleton axis as time."""
    x = np.squeeze(to_numpy(raw_x))
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    while x.ndim > 2:
        x = x[0]
    time_axis = int(np.argmax(x.shape))
    channel_axis = 1 - time_axis
    ch = 0 if channel is None else int(channel)
    ch = max(0, min(ch, x.shape[channel_axis] - 1))
    return np.take(x, ch, axis=channel_axis).reshape(-1)


def robust_scale(x: Any) -> float:
    """Return a MAD-based scale with a standard-deviation fallback."""
    a = to_numpy(x).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return 1.0
    med = np.median(a)
    scale = 1.4826 * float(np.median(np.abs(a - med)))
    if scale < EPS:
        scale = float(np.std(a))
    return max(scale, EPS)


def event_type(event: Mapping[str, Any]) -> str:
    """Read an event primitive name from common model/output field names."""
    for key in ("type", "primitive", "primitive_type", "event_type", "label", "name"):
        value = event.get(key)
        if value is not None and not isinstance(value, (list, tuple, dict, np.ndarray)):
            return str(value).lower()
    return "stable"


def event_interval(event: Mapping[str, Any], length: int) -> Tuple[int, int]:
    """Read and clamp an event interval; the returned end index is exclusive."""
    bounds = event.get("boundaries", event.get("interval", event.get("span")))
    if bounds is not None:
        a = to_numpy(bounds).reshape(-1)
        start, end = (float(a[0]), float(a[1])) if a.size >= 2 else (0.0, float(length))
    else:
        start = float(event.get("start", event.get("begin", 0)))
        end = float(event.get("end", event.get("stop", length)))
    # Model heads commonly emit normalized boundaries.
    if length > 2 and 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0:
        start, end = start * length, end * length
    start_i = max(0, min(int(round(start)), length - 1 if length else 0))
    end_i = max(start_i + 1, min(int(round(end)), length))
    return start_i, end_i


def event_confidence(event: Mapping[str, Any]) -> float:
    """Read a finite confidence and clamp it to ``[0, 1]``."""
    value = event.get("confidence", event.get("score", event.get("probability", 1.0)))
    try:
        result = float(to_numpy(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        result = 1.0
    return float(np.clip(result if np.isfinite(result) else 0.0, 0.0, 1.0))


def event_attributes(event: Mapping[str, Any]) -> Dict[str, float]:
    """Collect numeric event attributes from nested and flat dictionaries."""
    attrs: Dict[str, float] = {}
    nested = event.get("attributes", event.get("attrs", {}))
    if isinstance(nested, Mapping):
        source = {**event, **nested}
    else:
        source = dict(event)
    for key in ("amplitude", "slope", "duration", "delta_mean", "prominence",
                "frequency", "phase"):
        if key in source:
            try:
                attrs[key] = float(to_numpy(source[key]).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass
    return attrs


def _linear_slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    t = np.arange(y.size, dtype=np.float64)
    try:
        from scipy.stats import linregress

        return float(linregress(t, y).slope)
    except (ImportError, ValueError):
        return float(np.polyfit(t, y, 1)[0])


def _dominant_frequency(y: np.ndarray) -> Tuple[float, float]:
    """Return frequency in cycles/sample and sinusoidal phase."""
    if y.size < 4:
        return 0.0, 0.0
    centered = np.nan_to_num(y - np.nanmean(y))
    try:
        from scipy.signal import periodogram

        frequencies, power = periodogram(centered)
    except ImportError:
        spectrum = np.fft.rfft(centered)
        frequencies = np.fft.rfftfreq(centered.size)
        power = np.abs(spectrum) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) < EPS:
        return 0.0, 0.0
    index = int(np.argmax(power[1:]) + 1)
    frequency = float(frequencies[index])
    coefficient = np.sum(centered * np.exp(-2j * np.pi * frequency * np.arange(y.size)))
    return frequency, float(np.angle(coefficient))


def _peak_prominence(y: np.ndarray) -> Tuple[float, float]:
    """Return positive and negative peak prominences."""
    if y.size < 3:
        return 0.0, 0.0
    try:
        from scipy.signal import find_peaks, peak_prominences

        pos, _ = find_peaks(y)
        neg, _ = find_peaks(-y)
        pp = float(np.max(peak_prominences(y, pos)[0])) if pos.size else 0.0
        nprom = float(np.max(peak_prominences(-y, neg)[0])) if neg.size else 0.0
        return pp, nprom
    except ImportError:
        baseline = float(np.median(y))
        return float(np.max(y) - baseline), float(baseline - np.min(y))


def measure_attributes(
    raw_x: Any,
    event: Mapping[str, Any],
) -> Dict[str, float]:
    """Measure all V1 attributes on the event's raw, unnormalized samples."""
    channel = event.get("channel", event.get("channel_id"))
    x = signal_1d(raw_x, None if channel is None else int(channel))
    start, end = event_interval(event, x.size)
    y = x[start:end]
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        raise ValueError("event segment contains no finite samples")
    frequency, phase = _dominant_frequency(finite)
    pos_prom, neg_prom = _peak_prominence(finite)
    edge = max(1, min(finite.size // 3, max(2, finite.size // 5)))
    delta_mean = (
        float(np.mean(finite[-edge:]) - np.mean(finite[:edge]))
        if finite.size >= 2 else 0.0
    )
    baseline = float(np.median(finite))
    return {
        "amplitude": float(np.max(finite) - np.min(finite)),
        "slope": _linear_slope(finite),
        "duration": float(end - start),
        "delta_mean": delta_mean,
        "prominence": max(pos_prom, neg_prom),
        "positive_prominence": pos_prom,
        "negative_prominence": neg_prom,
        "frequency": frequency,
        "phase": phase,
        "mean": float(np.mean(finite)),
        "median": baseline,
        "std": float(np.std(finite)),
        "start": float(start),
        "end": float(end),
    }


