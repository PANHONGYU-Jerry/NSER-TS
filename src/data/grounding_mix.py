"""Mixed grounding dataset: SCM + real-background injection + raw real."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .common import TimeSeriesSample, padding_collate
from .controlled_scm import ControlledSCMDataset
from .real_background_injection import RealBackgroundInjectionDataset


@dataclass(frozen=True)
class GroundingMixConfig:
    scm_fraction: float = 0.4
    injection_fraction: float = 0.4
    raw_fraction: float = 0.2
    scm_size: int = 4000
    injection_size: int = 4000
    raw_size: int = 2000
    seq_len: int = 128
    pred_len: int = 16
    max_channels: int = 64
    data_root: str = "/data/panhongyu/TS-LLM/datasets"
    seed: int = 42
    smoke: bool = False


class GroundingMixDataset(Dataset[TimeSeriesSample]):
    """Sample from three grounding sources with fixed proportions."""

    SOURCES = ("scm", "injection", "raw")

    def __init__(self, config: GroundingMixConfig) -> None:
        if config.smoke:
            config = GroundingMixConfig(
                scm_fraction=config.scm_fraction,
                injection_fraction=config.injection_fraction,
                raw_fraction=config.raw_fraction,
                scm_size=16,
                injection_size=16,
                raw_size=8,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                max_channels=config.max_channels,
                data_root=config.data_root,
                seed=config.seed,
                smoke=True,
            )
        self.config = config
        total = config.scm_size + config.injection_size + config.raw_size
        self.size = total
        self.scm = ControlledSCMDataset(size=config.scm_size, split="train", seed=config.seed)
        self.injection = RealBackgroundInjectionDataset(
            size=config.injection_size,
            root=config.data_root,
            seed=config.seed + 17,
            seq_len=config.seq_len,
            max_channels=config.max_channels,
        )
        self.raw = RealBackgroundInjectionDataset(
            size=config.raw_size,
            root=config.data_root,
            seed=config.seed + 29,
            seq_len=config.seq_len,
            max_channels=config.max_channels,
            sources=("ETTm1", "Weather"),
        )
        self._schedule = self._build_schedule()

    def _build_schedule(self) -> list[str]:
        cfg = self.config
        counts = {
            "scm": cfg.scm_size,
            "injection": cfg.injection_size,
            "raw": cfg.raw_size,
        }
        schedule: list[str] = []
        for source, count in counts.items():
            schedule.extend([source] * count)
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        perm = torch.randperm(len(schedule), generator=generator).tolist()
        return [schedule[index] for index in perm]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TimeSeriesSample:
        source = self._schedule[index]
        if source == "scm":
            sample = self.scm[index % len(self.scm)]
            sample.task_label = {"source": "scm"}
            return sample
        if source == "injection":
            return self.injection[index % len(self.injection)]
        sample = self.raw[index % len(self.raw)]
        sample.events_gt = None
        sample.task_label = {"source": "raw"}
        return sample


def grounding_collate(batch: list[TimeSeriesSample | Mapping[str, Any]]) -> dict[str, Any]:
    collated = padding_collate(batch)
    events_list = []
    for item in batch:
        if isinstance(item, TimeSeriesSample):
            events_list.append(item.events_gt)
        else:
            events_list.append(item.get("events_gt"))
    collated["events_list"] = events_list
    collated["sources"] = [
        (item.task_label or {}).get("source", "unknown")
        if isinstance(item, TimeSeriesSample)
        else (item.get("task_label") or {}).get("source", "unknown")
        for item in batch
    ]
    return collated


def build_grounding_loader(
    config: GroundingMixConfig,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GroundingMixDataset(config)
    return DataLoader(
        dataset,
        batch_size=2 if config.smoke else batch_size,
        shuffle=shuffle,
        collate_fn=grounding_collate,
    )
