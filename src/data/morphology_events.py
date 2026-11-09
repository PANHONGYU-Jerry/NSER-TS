"""Rule-based morphology event supervision for Stage-1b decoder pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.event_schema import ATTRIBUTE_NAMES, PRIMITIVES

from .common import TimeSeriesSample, make_sample, padding_collate

DEFAULT_MORPHOLOGY_ROOTS: Dict[str, Path] = {
    "ucr": Path(
        "/data/panhongyu/TS-LLM/baseline/UCR/models_shape/generate_features/output"
    ),
    "classify": Path(
        "/data/panhongyu/TS-LLM/baseline/classify/models_shape/generate_features/output"
    ),
    "forecast": Path(
        "/data/panhongyu/TS-LLM/baseline/forecast/models_shape/generate_features/output"
    ),
    "tsb_ad": Path(
        "/data/panhongyu/TS-LLM/baseline/TSB-AD/models_shape/generate_features/output"
    ),
}

TRAIN_SPLITS = frozenset({"train", "val", "tuning"})
EXCLUDE_SPLITS = frozenset({"test", "eva"})

SHAPE_TO_PRIMITIVE: Dict[str, str] = {
    "stable": "stable",
    "trend_up": "trend_up",
    "trend_down": "trend_down",
    "shift_up": "level_shift_up",
    "shift_down": "level_shift_down",
    "spike": "spike",
    "valley": "valley",
    "periodic": "periodic",
    "frequency_change": "frequency_change",
    "level_shift_up": "level_shift_up",
    "level_shift_down": "level_shift_down",
}

ATTR_KEY_TO_INDEX: Dict[str, int] = {
    "amplitude": 0,
    "slope": 1,
    "rise_slope": 1,
    "fall_slope": 1,
    "total_change": 1,
    "duration": 2,
    "delta_mean": 3,
    "plateau_mean": 3,
    "prominence": 4,
    "frequency": 5,
    "phase": 6,
}

# Normalized attribute targets are unbounded when the robust scale of a
# near-constant channel collapses toward the 1e-6 floor, producing values up
# to ~1e6.  That explodes the event attribute loss and destabilizes training;
# clip to a range that keeps >97% of real signal while bounding the loss.
ATTRIBUTE_CLAMP = 10.0


@dataclass(frozen=True)
class MorphologySplitSpec:
    family: str
    dataset: str
    split: str
    split_dir: Path


def _primitive_id(event_type: str) -> int:
    primitive = SHAPE_TO_PRIMITIVE.get(str(event_type).strip().lower(), "stable")
    if primitive not in PRIMITIVES:
        primitive = "stable"
    return PRIMITIVES.index(primitive)


def _robust_scale(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64).ravel()
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(values))
    return max(scale, 1e-6)


def _normalize_boundary(value: float, length: int) -> float:
    if length <= 1:
        return 0.0
    if 0.0 <= float(value) <= 1.0:
        return float(value)
    return float(value) / float(length - 1)


def _event_attributes(
    attrs: Mapping[str, Any],
    *,
    length: int,
    scale: float,
    num_attributes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(num_attributes, dtype=torch.float32)
    mask = torch.zeros(num_attributes, dtype=torch.float32)
    for key, raw in (attrs or {}).items():
        index = ATTR_KEY_TO_INDEX.get(str(key))
        if index is None or index >= num_attributes:
            continue
        value = float(raw)
        if key in {"duration"}:
            value = value / max(length, 1)
        elif key in {"amplitude", "delta_mean", "plateau_mean", "prominence"}:
            value = value / scale
        elif key in {"slope", "rise_slope", "fall_slope", "total_change"}:
            value = value * length / scale
        values[index] = float(np.clip(value, -ATTRIBUTE_CLAMP, ATTRIBUTE_CLAMP))
        mask[index] = 1.0
    return values, mask


def labels_to_event_target(
    labels: Sequence[Mapping[str, Any]],
    *,
    length: int,
    raw_channel: np.ndarray,
    max_events: int,
    num_attributes: int = len(ATTRIBUTE_NAMES),
    channel_index: int = 0,
    n_channels: int = 1,
) -> Dict[str, torch.Tensor]:
    """Convert rule labels to EventSetCriterion target dict."""

    ordered = sorted(labels, key=lambda item: (int(item.get("start", 0)), int(item.get("id", 0))))
    if max_events > 0:
        ordered = ordered[:max_events]
    scale = _robust_scale(raw_channel)
    types: List[int] = []
    boundaries: List[List[float]] = []
    attributes: List[torch.Tensor] = []
    attribute_masks: List[torch.Tensor] = []
    for label in ordered:
        start = _normalize_boundary(float(label.get("start", 0)), length)
        end = _normalize_boundary(float(label.get("end", start)), length)
        if end < start:
            start, end = end, start
        types.append(_primitive_id(str(label.get("type", "stable"))))
        boundaries.append([start, end])
        attr_values, attr_mask = _event_attributes(
            label.get("attrs") or {},
            length=length,
            scale=scale,
            num_attributes=num_attributes,
        )
        attributes.append(attr_values)
        attribute_masks.append(attr_mask)
    count = len(types)
    if count == 0:
        return {
            "types": torch.zeros(0, dtype=torch.long),
            "boundaries": torch.zeros(0, 2, dtype=torch.float32),
            "channels": torch.zeros(0, max(n_channels, 1), dtype=torch.float32),
            "attributes": torch.zeros(0, num_attributes, dtype=torch.float32),
            "attribute_mask": torch.zeros(0, num_attributes, dtype=torch.float32),
            "event_channels": torch.zeros(0, dtype=torch.long),
            "confidence": torch.zeros(0, dtype=torch.float32),
        }
    confidence = torch.tensor(
        [float(label.get("confidence", 1.0)) for label in ordered],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    channel_one_hot = torch.zeros(count, max(n_channels, 1), dtype=torch.float32)
    channel_one_hot[:, min(channel_index, max(n_channels, 1) - 1)] = 1.0
    return {
        "types": torch.tensor(types, dtype=torch.long),
        "boundaries": torch.tensor(boundaries, dtype=torch.float32),
        "channels": channel_one_hot,
        "attributes": torch.stack(attributes),
        "attribute_mask": torch.stack(attribute_masks),
        "event_channels": torch.full((count,), int(channel_index), dtype=torch.long),
        "confidence": confidence,
    }


def _load_supervision_records(split_dir: Path) -> List[Dict[str, Any]]:
    path = split_dir / "multi_supervision.jsonl"
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_features_array(split_dir: Path, index: int, record: Mapping[str, Any]) -> np.ndarray:
    stacked = split_dir / "X_features.npy"
    if stacked.is_file():
        features = np.load(stacked, mmap_mode="r")
        return np.asarray(features[index], dtype=np.float32)
    feat_dir = split_dir / "features"
    file_name = str(record.get("file_name", ""))
    if not file_name:
        raise FileNotFoundError(f"Missing file_name for index {index} in {split_dir}")
    stem = Path(file_name).stem
    npz_path = feat_dir / f"{stem}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as payload:
        return np.asarray(payload["X_features"], dtype=np.float32)


def discover_morphology_splits(
    roots: Optional[Mapping[str, Path]] = None,
    *,
    include_splits: Optional[Iterable[str]] = None,
) -> List[MorphologySplitSpec]:
    allowed = set(include_splits or TRAIN_SPLITS)
    specs: List[MorphologySplitSpec] = []
    for family, root in (roots or DEFAULT_MORPHOLOGY_ROOTS).items():
        if not root.is_dir():
            continue
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for split_dir in sorted(dataset_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                split = split_dir.name
                if split in EXCLUDE_SPLITS or split not in allowed:
                    continue
                if not (split_dir / "multi_supervision.jsonl").exists():
                    continue
                specs.append(
                    MorphologySplitSpec(
                        family=family,
                        dataset=dataset_dir.name,
                        split=split,
                        split_dir=split_dir,
                    )
                )
    return specs


class MorphologyEventSplitDataset(Dataset[TimeSeriesSample]):
    """One generate_features split with aligned X_features and rule labels."""

    def __init__(
        self,
        spec: MorphologySplitSpec,
        *,
        max_events: int = 6,
        num_attributes: int = len(ATTRIBUTE_NAMES),
        use_raw_channel_only: bool = False,
    ) -> None:
        self.spec = spec
        self.max_events = int(max_events)
        self.num_attributes = int(num_attributes)
        self.use_raw_channel_only = bool(use_raw_channel_only)
        self.records = _load_supervision_records(spec.split_dir)
        if not self.records:
            raise ValueError(f"No supervision records in {spec.split_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TimeSeriesSample:
        record = self.records[index]
        features = _load_features_array(self.spec.split_dir, index, record)
        if features.ndim != 2:
            raise ValueError(f"Expected [T,C] features, got {features.shape}")
        length = int(features.shape[0])
        raw_channel = features[:, 0]
        if self.use_raw_channel_only:
            x = raw_channel[None, :]
        else:
            x = features.T
        n_channels = int(x.shape[0])
        labels = record.get("labels") or []
        events = labels_to_event_target(
            labels,
            length=length,
            raw_channel=raw_channel,
            max_events=self.max_events,
            num_attributes=self.num_attributes,
            n_channels=n_channels,
        )
        sample_id = (
            f"morph:{self.spec.family}:{self.spec.dataset}:"
            f"{self.spec.split}:{record.get('index', index)}"
        )
        return make_sample(
            x,
            {"source": sample_id, "n_events": int(events["types"].numel())},
            sample_id,
            events_gt=events,
        )


class MorphologyEventDataset(Dataset[TimeSeriesSample]):
    """Concatenated rule-labeled morphology supervision across task families."""

    def __init__(
        self,
        *,
        roots: Optional[Mapping[str, Path]] = None,
        include_splits: Optional[Iterable[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        max_events: int = 6,
        num_attributes: int = len(ATTRIBUTE_NAMES),
        use_raw_channel_only: bool = False,
        smoke: bool = False,
        max_samples_per_split: Optional[int] = None,
    ) -> None:
        specs = discover_morphology_splits(roots, include_splits=include_splits)
        if datasets:
            allowed = set(datasets)
            specs = [spec for spec in specs if spec.dataset in allowed]
        if not specs:
            raise FileNotFoundError("No morphology supervision splits were discovered.")
        parts: List[MorphologyEventSplitDataset] = []
        for spec in specs:
            dataset = MorphologyEventSplitDataset(
                spec,
                max_events=max_events,
                num_attributes=num_attributes,
                use_raw_channel_only=use_raw_channel_only,
            )
            if max_samples_per_split is not None and len(dataset) > max_samples_per_split:
                indices = np.linspace(0, len(dataset) - 1, max_samples_per_split, dtype=int)
                subset = torch.utils.data.Subset(dataset, indices.tolist())
                parts.append(subset)  # type: ignore[arg-type]
            else:
                parts.append(dataset)
        self._dataset: Dataset[TimeSeriesSample]
        if len(parts) == 1:
            self._dataset = parts[0]
        else:
            self._dataset = ConcatDataset(parts)
        if smoke and len(self._dataset) > 16:
            self._dataset = torch.utils.data.Subset(self._dataset, list(range(16)))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TimeSeriesSample:
        return self._dataset[index]


def build_morphology_event_loader(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    smoke: bool = False,
    max_events: int = 6,
    roots: Optional[Mapping[str, Path]] = None,
    datasets: Optional[Sequence[str]] = None,
    use_raw_channel_only: bool = False,
) -> DataLoader:
    dataset = MorphologyEventDataset(
        roots=roots,
        max_events=max_events,
        smoke=smoke,
        use_raw_channel_only=use_raw_channel_only,
        datasets=datasets,
    )
    return DataLoader(
        dataset,
        batch_size=2 if smoke else batch_size,
        shuffle=True,
        collate_fn=padding_collate,
        num_workers=num_workers,
    )


def make_morphology_event_dataset(**kwargs: Any) -> MorphologyEventDataset:
    return MorphologyEventDataset(**kwargs)


class MorphologyEventIndex:
    """Rule event labels keyed by (dataset, official-source) with alignment check.

    ``generate_features`` output may resample/augment official splits, so
    sample indices are only valid when verified against the raw channel.
    """

    def __init__(
        self,
        *,
        roots: Optional[Mapping[str, Path]] = None,
        max_events: int = 6,
        num_attributes: int = len(ATTRIBUTE_NAMES),
    ) -> None:
        self.max_events = int(max_events)
        self.num_attributes = int(num_attributes)
        self.roots = dict(DEFAULT_MORPHOLOGY_ROOTS if roots is None else roots)
        self._records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._raw: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        self._valid: Dict[Tuple[str, str], bool] = {}
        self._aligned_limit: Dict[Tuple[str, str], int] = {}

    def _locate(self, dataset: str, split: str) -> Optional[Path]:
        for root in self.roots.values():
            candidate = Path(root) / dataset / split
            if (candidate / "multi_supervision.jsonl").is_file():
                return candidate
        return None

    def records(self, dataset: str, split: str) -> Optional[List[Dict[str, Any]]]:
        key = (dataset, split)
        if key not in self._records:
            split_dir = self._locate(dataset, split)
            self._records[key] = (
                _load_supervision_records(split_dir) if split_dir is not None else []
            )
        return self._records[key] or None

    def raw_channel(self, dataset: str, split: str) -> Optional[np.ndarray]:
        """Raw-channel matrix [N,T] for a split (None when unavailable)."""
        key = (dataset, split)
        if key not in self._raw:
            split_dir = self._locate(dataset, split)
            if split_dir is None:
                self._raw[key] = None
            else:
                stacked = split_dir / "X_features.npy"
                if stacked.is_file():
                    features = np.load(stacked, mmap_mode="r")
                    self._raw[key] = np.asarray(features[..., 0], dtype=np.float32)
                else:
                    self._raw[key] = None
        return self._raw[key]

    def verify_alignment(
        self,
        dataset: str,
        official_x: np.ndarray,
        official_split: str = "train",
        *,
        num_check: int = 8,
    ) -> bool:
        """Check identity mapping ``official[source] == output[source]``.

        Verified against the raw channel; train/val use official TRAIN, test
        uses official TEST. Returns False when the output was resampled.
        """
        official = np.asarray(official_x, dtype=np.float32)
        official = official.reshape(official.shape[0], -1)
        out_split = "train" if official_split in ("train", "val") else "test"
        out_raw = self.raw_channel(dataset, out_split)
        key = (dataset, official_split)
        if out_raw is None or official.shape[1] != out_raw.shape[1]:
            self._valid[key] = False
            return False
        limit = min(len(official), len(out_raw))
        if limit <= 0:
            self._valid[key] = False
            return False
        rng = np.random.default_rng(0)
        chosen = rng.choice(limit, size=min(num_check, limit), replace=False)
        for index in chosen:
            if not np.allclose(
                official[index], out_raw[index], atol=1e-3, rtol=1e-4
            ):
                self._valid[key] = False
                return False
        self._valid[key] = True
        self._aligned_limit[key] = limit
        return True

    def aligned(self, dataset: str, split: str) -> bool:
        return self._valid.get((dataset, split), False)

    def aligned_limit(self, dataset: str, split: str) -> int:
        return self._aligned_limit.get((dataset, split), 0)

    def event_target(
        self,
        dataset: str,
        split: str,
        source: int,
        *,
        length: int,
        n_channels: int = 1,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.aligned(dataset, split):
            return None
        out_split = "train" if split in ("train", "val") else "test"
        recs = self.records(dataset, out_split)
        if recs is None or source >= len(recs):
            return None
        raw = self.raw_channel(dataset, out_split)
        raw_channel = raw[source] if raw is not None else np.zeros(max(length, 1), dtype=np.float32)
        return labels_to_event_target(
            recs[source].get("labels") or [],
            length=length,
            raw_channel=np.asarray(raw_channel, dtype=np.float32),
            max_events=self.max_events,
            num_attributes=self.num_attributes,
            n_channels=n_channels,
        )

    def rule_events(
        self, dataset: str, split: str, source: int, *, length: int
    ) -> List[Dict[str, Any]]:
        """Raw rule event records in primitive vocabulary (for evaluation)."""
        if not self.aligned(dataset, split):
            return []
        out_split = "train" if split in ("train", "val") else "test"
        recs = self.records(dataset, out_split)
        if recs is None or source >= len(recs):
            return []
        labels = recs[source].get("labels") or []
        result: List[Dict[str, Any]] = []
        for label in labels:
            start = _normalize_boundary(float(label.get("start", 0)), length)
            end = _normalize_boundary(float(label.get("end", start)), length)
            if end < start:
                start, end = end, start
            result.append(
                {
                    "type": PRIMITIVES[_primitive_id(str(label.get("type", "stable")))],
                    "boundary": [float(start), float(end)],
                }
            )
        return result
