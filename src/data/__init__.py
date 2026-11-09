"""Unified data layer for classification, detection, forecasting, and SCM data."""

from __future__ import annotations

from typing import Any

from .common import TimeSeriesSample, collate_fn, make_sample, padding_collate
from .controlled_scm import ControlledSCMDataset, EVENT_TYPES, make_controlled_scm
from .forecasting import ForecastingDataset, make_forecasting, make_forecasting_datasets
from .morphology_events import (
    MorphologyEventDataset,
    MorphologyEventIndex,
    build_morphology_event_loader,
    discover_morphology_splits,
    labels_to_event_target,
    make_morphology_event_dataset,
)
from .smd import SMDDataset, make_smd, make_smd_datasets
from .tsb_ad import TSBADDataset, make_tsb_ad, split_tsb_ad_tuning_files
from .uci_har import UCIHARDataset, make_uci_har, make_uci_har_datasets
from .ucr import UCR_DATASETS, UCRDataset, make_ucr, make_ucr_datasets
from .uea_ucr import (
    UEA_UCR_DATASETS,
    UEAUCRDataset,
    make_uea_ucr,
    make_uea_ucr_datasets,
)


def make_dataset(kind: str, **kwargs: Any):
    """Create any supported dataset through one small factory."""
    key = kind.lower().replace("-", "_")
    if key in ("controlled_scm", "scm"):
        return make_controlled_scm(**kwargs)
    if key == "ucr":
        return make_ucr(**kwargs)
    if key in ("uci_har", "har"):
        return make_uci_har(**kwargs)
    if key in ("smd", "server_machine", "servermachine"):
        return make_smd(**kwargs)
    if key in ("uea", "uea_ucr", "uea_ucr_event"):
        return make_uea_ucr(**kwargs)
    if key in ("tsb_ad", "ad"):
        return make_tsb_ad(**kwargs)
    if key in ("forecast", "forecasting", "ettm1", "weather"):
        if key in ("ettm1", "weather"):
            kwargs.setdefault("name", "ETTm1" if key == "ettm1" else "Weather")
        return make_forecasting(**kwargs)
    raise ValueError(f"Unknown dataset kind: {kind!r}")


__all__ = [
    "TimeSeriesSample",
    "padding_collate",
    "collate_fn",
    "make_sample",
    "ControlledSCMDataset",
    "EVENT_TYPES",
    "make_controlled_scm",
    "MorphologyEventDataset",
    "MorphologyEventIndex",
    "build_morphology_event_loader",
    "discover_morphology_splits",
    "labels_to_event_target",
    "make_morphology_event_dataset",
    "UCRDataset",
    "UCR_DATASETS",
    "make_ucr",
    "make_ucr_datasets",
    "UCIHARDataset",
    "make_uci_har",
    "make_uci_har_datasets",
    "SMDDataset",
    "make_smd",
    "make_smd_datasets",
    "UEAUCRDataset",
    "UEA_UCR_DATASETS",
    "make_uea_ucr",
    "make_uea_ucr_datasets",
    "TSBADDataset",
    "make_tsb_ad",
    "split_tsb_ad_tuning_files",
    "ForecastingDataset",
    "make_forecasting",
    "make_forecasting_datasets",
    "make_dataset",
]
