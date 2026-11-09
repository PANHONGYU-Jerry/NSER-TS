"""UEA event-classification adapter (gestures / activities).

Mirrors ``baseline/classify/data/uea_ucr.py`` so Stage-2 event classification
uses the same splits as the classify baseline.

Morphology-label UCR sets (Trace / MoteStrain / Coffee / ...) live under
``src.data.ucr`` and ``baseline/UCR/models_shape`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample

DEFAULT_ROOT = Path("/data/panhongyu/TS-LLM/datasets")

# Event / activity / gesture classification only.
UEA_UCR_DATASETS = (
    "BasicMotions",
    "UWaveGestureLibrary",
    "NATOPS",
    "Cricket",
    "Libras",
    "Epilepsy",
    "RacketSports",
)

CLASS_NAME_ORDER: Dict[str, List[str]] = {
    "BasicMotions": ["standing", "running", "walking", "badminton"],
    "UWaveGestureLibrary": [f"{i}.0" for i in range(1, 9)],
    "NATOPS": [f"{i}.0" for i in range(1, 7)],
    "Cricket": [f"{i}.0" for i in range(1, 13)],
    "Libras": [str(i) for i in range(1, 16)],
    # aeon normalizes these class strings on load.
    "Epilepsy": ["epilepsy", "walking", "running", "sawing"],
    "RacketSports": [
        "badminton_clear",
        "badminton_smash",
        "squash_backhandboast",
        "squash_forehandboast",
    ],
}

N_CLASSES: Dict[str, int] = {
    "BasicMotions": 4,
    "UWaveGestureLibrary": 8,
    "NATOPS": 6,
    "Cricket": 12,
    "Libras": 15,
    "Epilepsy": 4,
    "RacketSports": 4,
}


def is_uea_dataset(name: str | None) -> bool:
    return str(name) in UEA_UCR_DATASETS


def class_names_for(name: str) -> List[str]:
    return list(CLASS_NAME_ORDER.get(name, []))


def n_classes_for(name: str) -> int:
    return int(N_CLASSES.get(name, len(class_names_for(name)) or 2))


def resolve_uea_root(name: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Locate the dataset folder from either the datasets root or the folder."""
    root = Path(root).expanduser()
    if (root / f"{name}_TRAIN.ts").is_file():
        return root
    candidates = [
        root / name,
        Path("/data/panhongyu/TS-LLM/immigrant/datasets") / name,
    ]
    for candidate in candidates:
        if (candidate / f"{name}_TRAIN.ts").is_file() or candidate.exists():
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"UEA/UCR dataset folder not found for {name} under {root}")


def _ordered_class_names(dataset: str, labels: List[str]) -> List[str]:
    uniq = {str(v) for v in labels}
    preferred = CLASS_NAME_ORDER.get(dataset, [])
    ordered = [c for c in preferred if c in uniq]
    rest = sorted(uniq - set(ordered))
    return ordered + rest


def _remap_labels(
    y_train: np.ndarray, y_test: np.ndarray, dataset: str
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    names = _ordered_class_names(
        dataset, [str(v) for v in y_train.tolist()] + [str(v) for v in y_test.tolist()]
    )
    mapping = {name: i for i, name in enumerate(names)}
    y_tr = np.asarray([mapping[str(v)] for v in y_train], dtype=np.int64)
    y_te = np.asarray([mapping[str(v)] for v in y_test], dtype=np.int64)
    return y_tr, y_te, names, mapping


def _load_ts(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    from aeon.datasets import load_from_ts_file

    X, y = load_from_ts_file(path)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    elif X.ndim != 3:
        raise ValueError(f"Expected (N,C,T) or (N,T) in {path}, got {X.shape}")
    y = np.asarray(y)
    return X, y


def _zscore_channels(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / np.maximum(std, 1e-6)


class UEAUCRDataset(Dataset[TimeSeriesSample]):
    """One official UEA event split, with validation carved only from train."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalize: bool = True,
    ) -> None:
        if name not in UEA_UCR_DATASETS:
            raise ValueError(f"Unknown UEA/UCR dataset {name!r}; choices: {UEA_UCR_DATASETS}")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.name, self.split = name, split
        dataset_root = resolve_uea_root(name, root)
        X_tr, y_tr_raw = _load_ts(dataset_root / f"{name}_TRAIN.ts")
        X_te, y_te_raw = _load_ts(dataset_root / f"{name}_TEST.ts")
        y_tr, y_te, class_names, mapping = _remap_labels(y_tr_raw, y_te_raw, name)
        self.class_names = class_names
        self.class_to_index = mapping

        if normalize:
            # Per-channel stats from TRAIN only (N, C, T) -> mean/std over N,T
            mean = X_tr.mean(axis=(0, 2), keepdims=True)
            std = X_tr.std(axis=(0, 2), keepdims=True)
            X_tr = _zscore_channels(X_tr, mean, std)
            X_te = _zscore_channels(X_te, mean, std)

        if split == "test":
            self.x = np.asarray(X_te, dtype=np.float32)
            self.y = y_te
            self.source_indices = np.arange(len(y_te))
        else:
            from sklearn.model_selection import train_test_split

            idx = np.arange(len(y_tr))
            tr_idx, va_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=y_tr if len(np.unique(y_tr)) > 1 else None,
            )
            selected = tr_idx if split == "train" else va_idx
            self.x = np.asarray(X_tr[selected], dtype=np.float32)
            self.y = y_tr[selected]
            self.source_indices = selected

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> TimeSeriesSample:
        source = int(self.source_indices[index])
        return make_sample(
            self.x[index],
            torch.tensor(int(self.y[index]), dtype=torch.long),
            f"uea:{self.name}:{self.split}:{source}",
        )


def make_uea_ucr_datasets(
    name: str,
    root: str | Path = DEFAULT_ROOT,
    val_ratio: float = 0.2,
    seed: int = 42,
    normalize: bool = True,
) -> dict[str, UEAUCRDataset]:
    common = dict(name=name, root=root, val_ratio=val_ratio, seed=seed, normalize=normalize)
    return {split: UEAUCRDataset(split=split, **common) for split in ("train", "val", "test")}


def make_uea_ucr(name: str, split: str = "train", **kwargs: object) -> UEAUCRDataset:
    return UEAUCRDataset(name=name, split=split, **kwargs)


__all__ = [
    "UEA_UCR_DATASETS",
    "CLASS_NAME_ORDER",
    "N_CLASSES",
    "UEAUCRDataset",
    "is_uea_dataset",
    "class_names_for",
    "n_classes_for",
    "resolve_uea_root",
    "make_uea_ucr",
    "make_uea_ucr_datasets",
]
