"""Ground-truth event injection for the ``oracle_events`` ablation.

The oracle answers "how good would downstream classification be if the event
decoder were perfect?" by feeding rule ground-truth events straight into the
graph + task-head path.  Two sources are used:

* Datasets whose ``generate_features`` output is sample-aligned with the
  official UCR split (e.g. SyntheticControl) reuse the exact Stage-1b rule
  labels through :class:`MorphologyEventIndex`.
* Datasets whose generated output was resampled (e.g. CBF) have no aligned
  labels, so the same rule pipeline (pseudo interval + morphology
  decomposition, inlined from the baseline ``generate_features`` package) is
  re-run directly on the official signal so the label distribution matches
  Stage-1b supervision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.event_schema import ATTRIBUTE_NAMES, PRIMITIVES

from .morphology_events import (
    MorphologyEventIndex,
    labels_to_event_target,
)

# UCR CBF remapped 0-based class id -> geometric primitive.
UCR_CBF_PRIMITIVE = {0: "cylinder", 1: "bell", 2: "funnel"}
# Matches baseline generate_features.shape_taxonomy.
MIN_FG_SEGMENT_LEN = 4
# CBF body morphology by primitive.
_CBF_BODY_MORPH = {
    "cylinder": "stable",
    "bell": "trend_up",
    "funnel": "trend_down",
}


def _estimate_pseudo_interval(x: np.ndarray) -> Dict[str, Any]:
    """Pseudo interval for official UCR CBF (no start/end GT).

    Smooth -> baseline -> largest contiguous |x-baseline| region.  This
    replicates baseline ``cbf_supervision.estimate_pseudo_interval``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = len(x)
    win = max(3, T // 20)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    sm = np.convolve(x, kernel, mode="same")
    baseline = float(np.median(sm[: max(3, T // 10)]))
    score = np.abs(sm - baseline)
    thr = max(float(np.quantile(score, 0.55)), 0.15 * float(score.max() + 1e-8))
    mask = score >= thr
    best = (0, T - 1)
    if mask.any():
        best_len = -1
        i = 0
        while i < T:
            if mask[i]:
                j = i
                while j + 1 < T and mask[j + 1]:
                    j += 1
                if j - i > best_len:
                    best_len = j - i
                    best = (i, j)
                i = j + 1
            else:
                i += 1
    start, end = int(best[0]), int(best[1])
    seg = x[start : end + 1]
    amp = float(seg.max() - seg.min()) if len(seg) else 0.0
    peak_i = int(start + np.argmax(np.abs(seg - baseline))) if len(seg) else start
    rise = float((x[peak_i] - x[start]) / max(1, peak_i - start)) if peak_i > start else 0.0
    fall = float((x[end] - x[peak_i]) / max(1, end - peak_i)) if end > peak_i else 0.0
    area = (
        float(np.trapezoid(seg - baseline))
        if hasattr(np, "trapezoid")
        else float(np.trapz(seg - baseline))
    )
    return {
        "start": start,
        "end": end,
        "amplitude": amp,
        "height": float(np.mean(seg) - baseline) if len(seg) else 0.0,
        "rise_slope": rise,
        "fall_slope": fall,
        "area": area,
        "plateau_mean": float(np.mean(seg)) if len(seg) else 0.0,
    }


def _decompose_cbf_morphology(
    primitive: str,
    start: int,
    end: int,
    T: int,
    attrs: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Split one CBF active interval into morphological foreground segments.

    Replicates baseline ``cbf_morphology.decompose_cbf_morphology``:
    stable (baseline) -> shift_up (onset) -> {stable|trend_up|trend_down}
    (body) -> shift_down (offset) -> stable (baseline).
    """
    attrs = dict(attrs or {})
    prim = (primitive or "cylinder").strip()
    if prim not in UCR_CBF_PRIMITIVE.values():
        prim = "cylinder"
    start = max(0, int(start))
    end = min(T - 1, int(end))
    if end < start:
        start, end = 0, T - 1

    active_len = end - start + 1
    bw = max(2, min(active_len // 8, T // 25))
    amp = float(attrs.get("amplitude", attrs.get("height", 0.0)))
    rise = float(attrs.get("rise_slope", 0.0))
    fall = float(attrs.get("fall_slope", 0.0))
    delta = float(amp if amp > 0 else 6.0)

    def seg(
        stype: str, seg_start: int, seg_end: int, confidence: float, seg_attrs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        return {
            "type": stype,
            "start": int(seg_start),
            "end": int(seg_end),
            "confidence": float(confidence),
            "attrs": {"duration": float(seg_end - seg_start + 1), **(seg_attrs or {})},
        }

    out: List[Dict[str, Any]] = []
    if start >= MIN_FG_SEGMENT_LEN:
        out.append(seg("stable", 0, start - 1, 0.85))

    on_end = min(end, start + bw - 1)
    if on_end >= start:
        out.append(
            seg(
                "shift_up",
                start,
                on_end,
                0.95,
                {"delta_mean": delta, "boundary": "onset"},
            )
        )

    body_start = min(end, start + bw)
    body_end = max(body_start, end - bw)
    body_type = _CBF_BODY_MORPH[prim]
    body_attrs: Dict[str, float] = {}
    if body_type == "trend_up":
        body_attrs["slope"] = rise if abs(rise) > 1e-6 else delta / max(active_len, 1)
        body_attrs["total_change"] = body_attrs["slope"] * max(1, body_end - body_start)
    elif body_type == "trend_down":
        body_attrs["slope"] = fall if abs(fall) > 1e-6 else -delta / max(active_len, 1)
        body_attrs["total_change"] = body_attrs["slope"] * max(1, body_end - body_start)
    elif prim == "cylinder":
        body_attrs["plateau_mean"] = float(attrs.get("plateau_mean", delta))
    if body_end >= body_start:
        out.append(seg(body_type, body_start, body_end, 1.0, body_attrs))

    if prim == "bell" and active_len >= MIN_FG_SEGMENT_LEN + 2:
        peak_start = max(start, end - bw)
        if peak_start <= end:
            out.append(
                seg(
                    "spike",
                    peak_start,
                    end,
                    0.9,
                    {"amplitude": delta, "position": float(end)},
                )
            )

    off_start = max(start, end - bw + 1)
    if off_start <= end and prim != "bell":
        out.append(
            seg(
                "shift_down",
                off_start,
                end,
                0.95,
                {"delta_mean": -delta, "boundary": "offset"},
            )
        )

    if end + 1 <= T - MIN_FG_SEGMENT_LEN:
        out.append(seg("stable", end + 1, T - 1, 0.85))

    merged: List[Dict[str, Any]] = []
    for item in sorted(out, key=lambda s: s["start"]):
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        if item["type"] == prev["type"] and item["start"] <= prev["end"] + 1:
            prev["end"] = max(prev["end"], item["end"])
            prev["confidence"] = max(prev["confidence"], item["confidence"])
            prev["attrs"]["duration"] = float(prev["end"] - prev["start"] + 1)
        else:
            merged.append(item)
    return merged


def cbf_oracle_labels(
    x: np.ndarray, class_id: int, max_events: int
) -> List[Dict[str, Any]]:
    """Rule labels (Stage-1b vocabulary) for one official UCR CBF series."""
    x = np.asarray(x, dtype=np.float64).ravel()
    primitive = UCR_CBF_PRIMITIVE.get(int(class_id), "cylinder")
    est = _estimate_pseudo_interval(x)
    attrs = {key: value for key, value in est.items() if key not in ("start", "end")}
    segments = _decompose_cbf_morphology(primitive, est["start"], est["end"], len(x), attrs)
    labels: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if max_events > 0 and index >= max_events:
            break
        labels.append(
            {
                "type": segment["type"],
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "confidence": float(segment.get("confidence", 1.0)),
                "attrs": dict(segment.get("attrs") or {}),
                "layer": "foreground",
                "id": index,
            }
        )
    return labels


def _baseline_rule_labels(
    dataset: str,
    x: np.ndarray,
    class_id: int,
    max_events: int,
) -> List[Dict[str, Any]]:
    """Rule labels (Stage-1b vocabulary) for one official UCR series.

    Re-runs the baseline ``generate_features`` supervision + hierarchical
    multi-labeler directly on the official signal, so datasets whose generated
    output was resampled (all except SyntheticControl) still get oracle events
    that match Stage-1b supervision vocabulary.
    """
    try:
        import sys
        from pathlib import Path

        baseline_root = Path("/data/panhongyu/TS-LLM/baseline/UCR")
        shapes_root = baseline_root / "models_shape"
        for path in (str(shapes_root), str(baseline_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from generate_features.forda_supervision import build_forda_supervision
        from generate_features.multi_labeler import enrich_supervision_pack
        from generate_features.ops import extract_channels

        from generate_features.cbf_supervision import build_cbf_supervision
        from generate_features.ecg_supervision import build_ecg_supervision
        from generate_features.ecg5000_supervision import build_ecg5000_supervision
        from generate_features.wafer_supervision import build_wafer_supervision
    except ImportError:
        return []

    x = np.asarray(x, dtype=np.float64).ravel()
    try:
        if dataset == "CBF":
            pack = build_cbf_supervision(x, int(class_id), source="pseudo_interval")
        elif dataset == "TwoLeadECG":
            pack = build_ecg_supervision(x, int(class_id), source="pseudo_peak")
        elif dataset == "ECG5000":
            pack = build_ecg5000_supervision(x, int(class_id), source="pseudo_peak")
        elif dataset == "Wafer":
            pack = build_wafer_supervision(x, int(class_id), source="pseudo_anomaly")
        elif dataset in ("FordA", "FordB"):
            pack = build_forda_supervision(x, int(class_id), source="original_binary_only")
        elif dataset in (
            "ElectricDevices",
            "StarLightCurves",
            "Coffee",
            "ItalyPowerDemand",
            "Plane",
            "Trace",
            "MoteStrain",
            "TwoPatterns",
            "SyntheticControl",
        ):
            # Same wafer-style morphology pack used in baseline/UCR models_shape.
            pack = build_wafer_supervision(x, int(class_id), source=f"pseudo_{dataset.lower()}")
        else:
            return []
    except Exception:
        return []

    try:
        _, _, aux = extract_channels(x)
        enriched = enrich_supervision_pack(pack, x, aux)
        labels = (enriched.get("record") or {}).get("multi_labels", {}).get("labels") or []
    except Exception:
        return []
    result: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        if max_events > 0 and index >= max_events:
            break
        start = int(label.get("start", 0))
        end = int(label.get("end", start))
        if end < start:
            start, end = end, start
        result.append(
            {
                "type": str(label.get("type", "stable")),
                "start": start,
                "end": end,
                "confidence": float(label.get("confidence", 1.0)),
                "attrs": dict(label.get("attrs") or {}),
                "layer": str(label.get("layer", "foreground")),
                "id": index,
            }
        )
    return result


class OracleEventBuilder:
    """GT event targets for official UCR samples (aligned or rule-generated)."""

    def __init__(
        self,
        *,
        max_events: int = 6,
        num_attributes: int = len(ATTRIBUTE_NAMES),
        data_root: str = "/data/panhongyu/TS-LLM/datasets",
    ) -> None:
        self.max_events = int(max_events)
        self.num_attributes = int(num_attributes)
        self.data_root = data_root
        self._index = MorphologyEventIndex(
            max_events=self.max_events, num_attributes=self.num_attributes
        )
        self._cache: Dict[Tuple[str, str, int], Optional[Dict[str, torch.Tensor]]] = {}
        self._sample_cache: Dict[str, Optional[Dict[str, torch.Tensor]]] = {}

    def verify(self, dataset: str, official_x: np.ndarray, official_split: str = "train") -> bool:
        """Verify generate_features alignment for a dataset/split."""
        return self._index.verify_alignment(dataset, official_x, official_split)

    def aligned(self, dataset: str, split: str) -> bool:
        return self._index.aligned(dataset, split)

    def event_target(
        self,
        dataset: str,
        split: str,
        source: int,
        *,
        length: int,
        n_channels: int = 1,
        raw_x: Optional[np.ndarray] = None,
        class_id: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Oracle GT event target for one official UCR sample (None = n/a).

        Resolution order:
          1. aligned Stage-1b rule labels (SyntheticControl);
          2. CBF inline rule decomposition;
          3. baseline generate_features rule pipeline re-run on the official
             signal (ECG5000 / TwoLeadECG / Wafer / FordA).
        """
        cache_key = (dataset, split, int(source))
        if cache_key in self._cache:
            return self._cache[cache_key]

        target = self._index.event_target(
            dataset, split, source, length=length, n_channels=n_channels
        )
        if target is None:
            if raw_x is not None:
                labels = cbf_oracle_labels(
                    raw_x, int(class_id or 0), self.max_events
                ) if dataset == "CBF" else _baseline_rule_labels(
                    dataset, raw_x, int(class_id or 0), self.max_events
                )
                if labels:
                    target = labels_to_event_target(
                        labels,
                        length=length,
                        raw_channel=np.asarray(raw_x, dtype=np.float32),
                        max_events=self.max_events,
                        num_attributes=self.num_attributes,
                        n_channels=n_channels,
                    )
        self._cache[cache_key] = target
        return target

    def event_target_for_sample(
        self,
        sample_id: str,
        x: np.ndarray,
        *,
        class_id: Optional[int] = None,
        channel_mask: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Oracle EventSet for any Stage-2 sample_id family.

        Supported prefixes:
          - ``ucr:{dataset}:{split}:{source}`` — morphology (aligned / rule)
          - ``uci_har:...`` — body-acc magnitude auto morphology
          - ``uea:{dataset}:...`` — accelerometer magnitude / Trace auto morphology
          - ``smd:...`` — auto morphology on highest-variance channels
          - ``tsb_ad:...`` — auto morphology on highest-variance channels
          - ``forecast:...`` — auto morphology on highest-variance channels
        """
        sample_id = str(sample_id)
        if sample_id in self._sample_cache:
            return self._sample_cache[sample_id]

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        length = int(x_arr.shape[-1])
        n_channels = int(x_arr.shape[0])
        target: Optional[Dict[str, torch.Tensor]] = None

        parts = sample_id.split(":")
        prefix = parts[0] if parts else ""
        if prefix == "ucr" and len(parts) >= 4:
            try:
                source = int(parts[3])
            except ValueError:
                source = -1
            target = self.event_target(
                parts[1],
                parts[2],
                source,
                length=length,
                n_channels=1,
                raw_x=x_arr[0],
                class_id=class_id,
            )
        elif prefix == "uci_har":
            labels = _har_baseline_labels(
                x_arr, int(class_id if class_id is not None else 0), self.max_events
            )
            if not labels:
                labels = _har_auto_labels(x_arr, self.max_events)
            if labels:
                mag = _body_acc_magnitude(x_arr)
                target = labels_to_event_target(
                    labels,
                    length=length,
                    raw_channel=np.asarray(mag, dtype=np.float32),
                    max_events=self.max_events,
                    num_attributes=self.num_attributes,
                    n_channels=n_channels,
                    channel_index=0,
                )
        elif prefix == "uea":
            # Prefer classify baseline build_uea_supervision (+ multi-labeler),
            # same grade as HAR _har_baseline_labels. Fall back to auto rules
            # on the dataset-specific univariate reduction.
            dataset = parts[1] if len(parts) >= 2 else ""
            series = _uea_teacher_series(x_arr, dataset)
            labels = _uea_baseline_labels(
                x_arr,
                dataset,
                int(class_id if class_id is not None else 0),
                self.max_events,
            )
            if not labels:
                labels = _auto_shape_labels(series, self.max_events)
            if labels:
                target = labels_to_event_target(
                    labels,
                    length=length,
                    raw_channel=np.asarray(series, dtype=np.float32),
                    max_events=self.max_events,
                    num_attributes=self.num_attributes,
                    n_channels=n_channels,
                    channel_index=0,
                )
        elif prefix == "smd":
            # Prefer classify baseline build_smd_supervision; fall back to
            # multi-channel auto morphology on highest-variance channels.
            labels = _smd_baseline_labels(
                x_arr,
                int(class_id if class_id is not None else 0),
                self.max_events,
            )
            if labels:
                series = _smd_teacher_series(x_arr)
                target = labels_to_event_target(
                    labels,
                    length=length,
                    raw_channel=np.asarray(series, dtype=np.float32),
                    max_events=self.max_events,
                    num_attributes=self.num_attributes,
                    n_channels=n_channels,
                    channel_index=0,
                )
            else:
                labels = _multichannel_auto_labels(
                    x_arr,
                    self.max_events,
                    channel_mask=channel_mask,
                    max_channels=2,
                )
                if labels:
                    target = _labels_with_channels_to_target(
                        labels,
                        length=length,
                        x_ct=x_arr,
                        max_events=self.max_events,
                        num_attributes=self.num_attributes,
                        n_channels=n_channels,
                    )
        elif prefix == "tsb_ad":
            target = _tsb_ad_label_target(
                sample_id,
                x_arr,
                max_events=self.max_events,
                num_attributes=self.num_attributes,
                n_channels=n_channels,
                channel_mask=channel_mask,
                data_root=self.data_root,
            )
            if target is None:
                labels = _multichannel_auto_labels(
                    x_arr,
                    self.max_events,
                    channel_mask=channel_mask,
                    max_channels=2,
                )
                if labels:
                    target = _labels_with_channels_to_target(
                        labels,
                        length=length,
                        x_ct=x_arr,
                        max_events=self.max_events,
                        num_attributes=self.num_attributes,
                        n_channels=n_channels,
                    )
        elif prefix == "forecast":
            labels = _multichannel_auto_labels(
                x_arr,
                self.max_events,
                channel_mask=channel_mask,
                max_channels=3,
            )
            if labels:
                target = _labels_with_channels_to_target(
                    labels,
                    length=length,
                    x_ct=x_arr,
                    max_events=self.max_events,
                    num_attributes=self.num_attributes,
                    n_channels=n_channels,
                )

        self._sample_cache[sample_id] = target
        return target


def _ensure_ucr_generate_features_path() -> None:
    import sys
    from pathlib import Path

    baseline_root = Path("/data/panhongyu/TS-LLM/baseline/UCR")
    shapes_root = baseline_root / "models_shape"
    for path in (str(shapes_root), str(baseline_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_CLASSIFY_GEN_DIR = "/data/panhongyu/TS-LLM/baseline/classify/models_shape/generate_features"
_UCR_GEN_DIR = "/data/panhongyu/TS-LLM/baseline/UCR/models_shape/generate_features"


def _load_classify_gen_module(name: str) -> Any:
    """Load a classify-side ``generate_features`` submodule (e.g.
    ``uci_har_supervision``) into the shared ``generate_features`` package.

    The classify folder reuses the UCR ``generate_features`` package namespace
    via ``_ucr_deps``; loading the classify-only modules this way keeps
    relative imports (``from ._ucr_deps import ucr_modules``) working while the
    package ``__path__`` points at the UCR implementation.
    """
    import importlib.util
    import sys
    import types
    from pathlib import Path

    _ensure_ucr_generate_features_path()
    if "generate_features" not in sys.modules:
        pkg = types.ModuleType("generate_features")
        pkg.__path__ = [str(Path(_UCR_GEN_DIR))]
        pkg.__package__ = "generate_features"
        sys.modules["generate_features"] = pkg
    full = f"generate_features.{name}"
    if full in sys.modules:
        return sys.modules[full]
    path = Path(_CLASSIFY_GEN_DIR) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(full, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "generate_features"
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _auto_shape_labels(x_1d: np.ndarray, max_events: int) -> List[Dict[str, Any]]:
    """Stage-1b vocabulary labels via UCR hierarchical multi-labeler."""
    try:
        _ensure_ucr_generate_features_path()
        from generate_features.multi_labeler import build_multi_shape_labels
        from generate_features.ops import extract_channels
    except ImportError:
        return []
    x = np.asarray(x_1d, dtype=np.float64).ravel()
    if x.size == 0:
        return []
    try:
        _, _, aux = extract_channels(x)
        multi = build_multi_shape_labels(x, aux, gt_hint=None, max_labels=max(1, max_events))
        labels = (multi.get("record") or {}).get("labels") or []
    except Exception:
        return []
    result: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        if max_events > 0 and index >= max_events:
            break
        start = int(label.get("start", 0))
        end = int(label.get("end", start))
        if end < start:
            start, end = end, start
        result.append(
            {
                "type": str(label.get("type", "stable")),
                "start": start,
                "end": end,
                "confidence": float(label.get("confidence", 1.0)),
                "attrs": dict(label.get("attrs") or {}),
                "layer": str(label.get("layer", "foreground")),
                "id": index,
            }
        )
    return result


def _body_acc_magnitude(x_ct: np.ndarray) -> np.ndarray:
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        return x.ravel()
    if x.shape[0] >= 3:
        return np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    return np.asarray(x[0], dtype=np.float64).ravel()


def _uea_default_channel_method(dataset: str) -> str:
    """Mirror baseline ``uea_series.default_channel_method`` (no import path risk)."""
    name = str(dataset)
    if name in (
        "BasicMotions",
        "UWaveGestureLibrary",
        "Cricket",
        "NATOPS",
        "Libras",
        "Epilepsy",
        "RacketSports",
    ):
        return "mag"
    return "max_var"


def _uea_to_univariate(x_raw: np.ndarray, method: str) -> np.ndarray:
    """Local copy of baseline ``uea_series.to_univariate`` core paths."""
    method = (method or "max_var").lower()
    x = np.asarray(x_raw, dtype=np.float64)
    if x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
        return x.ravel()
    if x.ndim != 2:
        raise ValueError(f"Expected (C,T) or (T,), got {x.shape}")
    # Prefer (C, T) layout used by model loaders.
    if x.shape[0] > x.shape[1]:
        # Likely (T, C); transpose to (C, T).
        x = x.T
    if method in ("mag", "accel_mag", "magnitude"):
        axes = x[: min(3, x.shape[0])]
        return np.sqrt((axes ** 2).sum(axis=0))
    if method in ("rms", "rms_mag"):
        return np.sqrt((x ** 2).mean(axis=0))
    if method.startswith("dim"):
        digits = "".join(ch for ch in method if ch.isdigit())
        idx = int(digits) if digits else 0
        idx = min(idx, x.shape[0] - 1)
        return x[idx].ravel()
    var = x.var(axis=1)
    idx = int(np.argmax(var))
    return x[idx].ravel()


def _uea_teacher_series(x_ct: np.ndarray, dataset: str) -> np.ndarray:
    """Univariate channel the classify teacher labels were generated on."""
    method = _uea_default_channel_method(dataset)
    return np.asarray(_uea_to_univariate(x_ct, method), dtype=np.float64).ravel()


def _smd_teacher_series(x_ct: np.ndarray) -> np.ndarray:
    return np.asarray(_uea_to_univariate(x_ct, "max_var"), dtype=np.float64).ravel()


def _har_auto_labels(x_ct: np.ndarray, max_events: int) -> List[Dict[str, Any]]:
    """Auto morphology on UCI HAR body-acceleration magnitude."""
    return _auto_shape_labels(_body_acc_magnitude(x_ct), max_events)


def _labels_from_enriched_pack(
    pack: Mapping[str, Any], max_events: int
) -> List[Dict[str, Any]]:
    """Convert an enriched supervision pack into oracle/aux event dicts."""
    labels = (pack.get("record") or {}).get("multi_labels", {}).get("labels") or []
    result: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        if max_events > 0 and index >= max_events:
            break
        start = int(label.get("start", 0))
        end = int(label.get("end", start))
        if end < start:
            start, end = end, start
        result.append(
            {
                "type": str(label.get("type", "stable")),
                "start": start,
                "end": end,
                "confidence": float(label.get("confidence", 1.0)),
                "attrs": dict(label.get("attrs") or {}),
                "layer": str(label.get("layer", "foreground")),
                "id": index,
            }
        )
    return result


def _enrich_supervision_pack(
    pack: Mapping[str, Any], max_events: int
) -> Mapping[str, Any]:
    from generate_features.ops import extract_channels
    from generate_features.multi_labeler import enrich_supervision_pack

    _, _, aux = extract_channels(pack["x"])
    return enrich_supervision_pack(
        pack, pack["x"], aux, max_labels=max(1, max_events)
    )


def _ensure_classify_root_on_path() -> None:
    import sys
    from pathlib import Path

    classify_root = Path("/data/panhongyu/TS-LLM/baseline/classify")
    if str(classify_root) not in sys.path:
        sys.path.insert(0, str(classify_root))


def _har_baseline_labels(
    x_ct: np.ndarray, activity_id: int, max_events: int
) -> List[Dict[str, Any]]:
    """Stage-1b vocabulary labels for one official UCI HAR window.

    Runs the same supervision path as the baseline ``generate_features``
    ``original`` pipeline: ``build_har_supervision`` (weak pseudo_har labels on
    body-acc magnitude) followed by the hierarchical multi-labeler enrichment,
    so oracle events are byte-for-byte consistent with baseline labelling
    (unlike :func:`_har_auto_labels`, which re-runs generic auto rules).
    """
    try:
        _load_classify_gen_module("_ucr_deps")
        _load_classify_gen_module("har_series")
        uci_har_supervision = _load_classify_gen_module("uci_har_supervision")
    except ImportError:
        return []
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    try:
        pack = uci_har_supervision.build_har_supervision(
            x, int(activity_id or 0), source="pseudo_har", channel="body_acc_mag"
        )
        enriched = _enrich_supervision_pack(pack, max_events)
        return _labels_from_enriched_pack(enriched, max_events)
    except Exception:
        return []


def _uea_baseline_labels(
    x_ct: np.ndarray, dataset: str, class_id: int, max_events: int
) -> List[Dict[str, Any]]:
    """Classify-baseline UEA morphology labels (pseudo_uea + multi-labeler)."""
    try:
        _ensure_classify_root_on_path()
        _load_classify_gen_module("_ucr_deps")
        _load_classify_gen_module("uea_series")
        uea_supervision = _load_classify_gen_module("uea_supervision")
    except ImportError:
        return []
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    method = _uea_default_channel_method(dataset)
    try:
        pack = uea_supervision.build_uea_supervision(
            x,
            int(class_id or 0),
            dataset=str(dataset),
            channel_method=method,
            source=f"pseudo_{str(dataset).lower()}",
        )
        enriched = _enrich_supervision_pack(pack, max_events)
        return _labels_from_enriched_pack(enriched, max_events)
    except Exception:
        return []


def _smd_baseline_labels(
    x_ct: np.ndarray, class_id: int, max_events: int
) -> List[Dict[str, Any]]:
    """Classify-baseline SMD morphology labels (pseudo_smd + multi-labeler)."""
    try:
        _ensure_classify_root_on_path()
        _load_classify_gen_module("_ucr_deps")
        _load_classify_gen_module("smd_series")
        smd_supervision = _load_classify_gen_module("smd_supervision")
    except ImportError:
        return []
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    try:
        pack = smd_supervision.build_smd_supervision(
            x,
            int(class_id or 0),
            channel_method="max_var",
            source="pseudo_smd",
        )
        enriched = _enrich_supervision_pack(pack, max_events)
        return _labels_from_enriched_pack(enriched, max_events)
    except Exception:
        return []


def _active_channel_indices(
    x_ct: np.ndarray,
    *,
    channel_mask: Optional[np.ndarray] = None,
    max_channels: int = 3,
) -> List[int]:
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        return [0]
    n = int(x.shape[0])
    active = list(range(n))
    if channel_mask is not None:
        mask = np.asarray(channel_mask).astype(bool).ravel()
        active = [i for i in active if i < len(mask) and bool(mask[i])]
    if not active:
        active = [0]
    ranked = sorted(active, key=lambda i: float(np.var(x[i])), reverse=True)
    return ranked[: max(1, int(max_channels))]


def _multichannel_auto_labels(
    x_ct: np.ndarray,
    max_events: int,
    *,
    channel_mask: Optional[np.ndarray] = None,
    max_channels: int = 3,
) -> List[Dict[str, Any]]:
    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    merged: List[Dict[str, Any]] = []
    per_channel_budget = max(1, int(max_events))
    for channel in _active_channel_indices(
        x, channel_mask=channel_mask, max_channels=max_channels
    ):
        for label in _auto_shape_labels(x[channel], per_channel_budget):
            row = dict(label)
            row["_channel"] = int(channel)
            merged.append(row)
    merged.sort(key=lambda item: -float(item.get("confidence", 0.0)))
    if max_events > 0:
        merged = merged[:max_events]
    for index, row in enumerate(merged):
        row["id"] = index
    return merged


def _tsb_ad_label_target(
    sample_id: str,
    x_ct: np.ndarray,
    *,
    max_events: int,
    num_attributes: int,
    n_channels: int,
    channel_mask: Optional[np.ndarray],
    data_root: str = "/data/panhongyu/TS-LLM/datasets",
) -> Optional[Dict[str, torch.Tensor]]:
    """Oracle events from official TSB-AD point Labels for one window.

    sample_id layout: ``tsb_ad:{modality}:{split}:{filename}:{w0}:{w1}``.
    Each labelled anomalous interval becomes a foreground event typed by the
    window-local signal: a short high-amplitude burst is a ``spike``, longer
    offsets are ``level_shift_up`` / ``level_shift_down`` by sign of the mean
    shift.  Falls back to None so callers can fall back to auto rules.
    """
    from pathlib import Path

    parts = sample_id.split(":")
    if len(parts) != 6:
        return None
    modality, filename = parts[1], parts[3]
    try:
        w0, w1 = int(parts[4]), int(parts[5])
    except ValueError:
        return None
    series_root = Path(data_root) / f"TSB-AD-{modality.upper()}"
    path = series_root / filename
    try:
        import pandas as pd

        labels = pd.read_csv(path, usecols=["Label"]).to_numpy(dtype=np.int64).ravel()
    except Exception:
        return None
    seg = labels[w0:w1]
    length = int(seg.shape[0])
    if length == 0:
        return None
    active = seg.astype(bool)
    if not active.any():
        return None
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    channels = _active_channel_indices(x, channel_mask=channel_mask, max_channels=1)
    channel = int(channels[0]) if channels else 0
    trace = x[min(channel, x.shape[0] - 1)]
    global_med = float(np.median(trace))

    out: List[Dict[str, Any]] = []
    for index, (s, e) in enumerate(zip(starts, ends)):
        if max_events > 0 and index >= max_events:
            break
        s = min(max(int(s), 0), length - 1)
        e = min(max(int(e), s + 1), length)
        seg_trace = trace[s:e]
        amp = float(np.max(np.abs(seg_trace - global_med))) if seg_trace.size else 0.0
        mean_delta = float(np.mean(seg_trace) - global_med) if seg_trace.size else 0.0
        width = e - s
        if width <= max(2, int(0.1 * length)) and amp > 1e-6:
            etype = "spike"
            attrs: Dict[str, float] = {
                "amplitude": amp,
                "duration": float(width),
                "position": float(s + width / 2.0),
            }
        else:
            etype = "level_shift_up" if mean_delta >= 0 else "level_shift_down"
            attrs = {"delta_mean": mean_delta, "duration": float(width), "boundary": "onset"}
        out.append(
            {
                "type": etype,
                "start": s,
                "end": e - 1,
                "confidence": 1.0,
                "attrs": dict(attrs),
                "layer": "foreground",
                "_channel": channel,
                "id": index,
            }
        )
    if not out:
        return None
    return _labels_with_channels_to_target(
        out,
        length=length,
        x_ct=x,
        max_events=max_events,
        num_attributes=num_attributes,
        n_channels=n_channels,
    )


def _labels_with_channels_to_target(
    labels: Sequence[Mapping[str, Any]],
    *,
    length: int,
    x_ct: np.ndarray,
    max_events: int,
    num_attributes: int,
    n_channels: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if not labels:
        return None
    x = np.asarray(x_ct, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    by_channel: Dict[int, List[Mapping[str, Any]]] = {}
    for label in labels:
        channel = int(label.get("_channel", 0))
        by_channel.setdefault(channel, []).append(label)
    parts: List[Dict[str, torch.Tensor]] = []
    for channel, channel_labels in by_channel.items():
        raw = x[min(channel, x.shape[0] - 1)]
        parts.append(
            labels_to_event_target(
                channel_labels,
                length=length,
                raw_channel=raw,
                max_events=max_events,
                num_attributes=num_attributes,
                n_channels=n_channels,
                channel_index=min(channel, max(n_channels, 1) - 1),
            )
        )
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {
        "types": torch.cat([part["types"] for part in parts], dim=0),
        "boundaries": torch.cat([part["boundaries"] for part in parts], dim=0),
        "channels": torch.cat([part["channels"] for part in parts], dim=0),
        "attributes": torch.cat([part["attributes"] for part in parts], dim=0),
        "attribute_mask": torch.cat([part["attribute_mask"] for part in parts], dim=0),
        "event_channels": torch.cat([part["event_channels"] for part in parts], dim=0),
        "confidence": torch.cat([part["confidence"] for part in parts], dim=0),
    }
