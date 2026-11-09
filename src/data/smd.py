"""SMD (ServerMachineDataset) windowed adapter for event classification.

Mirrors the ``baseline/classify/scripts/run_all.sh`` data protocol
(``baseline/classify/data/smd.py``):

* every ``machine-*.txt`` is one entity (like a HAR subject)
* sliding windows (``seq_len=100``, ``stride=50``); a window is class 1 when
  any point inside it is anomalous, 0 otherwise
* machine-wise train/val/test split (never an i.i.d. window shuffle)
* train machines: official ``train`` windows (all normal) + official ``test``
  windows (point-labelled); val/test machines: official test windows only
* per-channel z-score normalization from train-window statistics

The event-classification head therefore learns a binary normal/anomaly window
task with ``num_event_classes=2``, index-aligned with the classify baseline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import TimeSeriesSample, make_sample


DEFAULT_ROOT = Path(
    os.environ.get(
        "SMD_ROOT",
        "/data/panhongyu/TS-LLM/datasets/OmniAnomaly-master/ServerMachineDataset",
    )
)
CLASS_NAMES = ["normal", "anomaly"]
N_CLASSES = 2
DEFAULT_SEQ_LEN = 100
DEFAULT_STRIDE = 50
DEFAULT_TEST_RATIO = 0.2


def resolve_smd_root(data_root: str | Path) -> Path:
    """Point at the SMD folder from either the datasets root or the SMD dir."""
    root = Path(data_root).expanduser()
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root
    candidate = root / "OmniAnomaly-master" / "ServerMachineDataset"
    if not candidate.exists():
        raise FileNotFoundError(
            f"SMD data not found under {root!s} (expected a ServerMachineDataset "
            f"folder or its datasets root)"
        )
    return candidate


def _list_machines(root: Path) -> List[str]:
    names = sorted(
        p.stem for p in (root / "train").glob("machine-*.txt") if p.is_file()
    )
    if not names:
        raise FileNotFoundError(f"No machine-*.txt under {root / 'train'}")
    return names


def _load_series(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr  # (T, C)


def _load_labels(path: Path) -> np.ndarray:
    y = np.genfromtxt(path, dtype=np.float32)
    return (y.reshape(-1) > 0.5).astype(np.int64)


def _sliding_windows(
    series: np.ndarray,
    labels: Optional[np.ndarray],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_raw [N,C,T], y [N])`` for one machine."""
    t, c = series.shape
    if t < seq_len:
        return np.zeros((0, c, seq_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for start in range(0, t - seq_len + 1, stride):
        end = start + seq_len
        xs.append(series[start:end].T)
        if labels is None:
            ys.append(0)
        else:
            ys.append(1 if labels[start:end].astype(np.int64).any() else 0)
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
    )


def _machine_val_split(
    machines: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[List[str], List[str], List[str]]:
    names = list(machines)
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
    test = names[:n_test]
    val = names[n_test : n_test + n_val]
    train = names[n_test + n_val :]
    assert train and val and test
    return train, val, test


def _collect_machine_windows(
    root: Path,
    machines: Sequence[str],
    seq_len: int,
    stride: int,
    include_train_normals: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate windows over machines -> ``(X [N,C,T], y [N], machine [N])``."""
    raws: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    names: List[np.ndarray] = []
    for m in machines:
        test_x = _load_series(root / "test" / f"{m}.txt")
        test_y = _load_labels(root / "test_label" / f"{m}.txt")
        if len(test_y) != len(test_x):
            raise ValueError(f"{m}: test length {len(test_x)} != label length {len(test_y)}")
        x_te, y_te = _sliding_windows(test_x, test_y, seq_len, stride)
        if len(y_te):
            raws.append(x_te)
            labels.append(y_te)
            names.append(np.array([m] * len(y_te), dtype=object))
        if include_train_normals:
            train_x = _load_series(root / "train" / f"{m}.txt")
            x_tr, y_tr = _sliding_windows(train_x, None, seq_len, stride)
            if len(y_tr):
                raws.append(x_tr)
                labels.append(y_tr)
                names.append(np.array([m] * len(y_tr), dtype=object))
    if not raws:
        return (
            np.zeros((0, 38, seq_len), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.array([], dtype=object),
        )
    return (
        np.concatenate(raws, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(names, axis=0),
    )


def _load_storage(
    root: Path,
    seq_len: int,
    stride: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    normalize: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    if not root.exists():
        raise FileNotFoundError(f"SMD root not found: {root}")
    machines = _list_machines(root)
    tr_m, va_m, te_m = _machine_val_split(
        machines, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    X_tr, y_tr, n_tr = _collect_machine_windows(
        root, tr_m, seq_len, stride, include_train_normals=True
    )
    X_va, y_va, n_va = _collect_machine_windows(
        root, va_m, seq_len, stride, include_train_normals=False
    )
    X_te, y_te, n_te = _collect_machine_windows(
        root, te_m, seq_len, stride, include_train_normals=False
    )
    if normalize and len(X_tr):
        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
        X_tr = (X_tr - mean) / std
        if len(X_va):
            X_va = (X_va - mean) / std
        if len(X_te):
            X_te = (X_te - mean) / std
    return {
        "train": {"x": X_tr, "y": y_tr, "machine": n_tr},
        "val": {"x": X_va, "y": y_va, "machine": n_va},
        "test": {"x": X_te, "y": y_te, "machine": n_te},
    }


class SMDDataset(Dataset[TimeSeriesSample]):
    def __init__(
        self,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = DEFAULT_STRIDE,
        val_ratio: float = 0.2,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = 42,
        normalize: bool = True,
        _storage: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        storage = _storage or _load_storage(
            resolve_smd_root(root),
            seq_len=seq_len,
            stride=stride,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            normalize=normalize,
        )
        part = storage[split]
        self.x: np.ndarray = part["x"]
        self.y: np.ndarray = part["y"]
        self.machine_names: np.ndarray = part["machine"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> TimeSeriesSample:
        return make_sample(
            self.x[index],
            torch.tensor(int(self.y[index]), dtype=torch.long),
            f"smd:{self.split}:{self.machine_names[index]}:{index}",
        )


def make_smd_datasets(
    root: str | Path = DEFAULT_ROOT,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    val_ratio: float = 0.2,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    normalize: bool = True,
) -> Dict[str, SMDDataset]:
    storage = _load_storage(
        resolve_smd_root(root),
        seq_len=seq_len,
        stride=stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
    )
    return {
        split: SMDDataset(split=split, _storage=storage)
        for split in ("train", "val", "test")
    }


def make_smd(split: str = "train", **kwargs: object) -> SMDDataset:
    return SMDDataset(split=split, **kwargs)
