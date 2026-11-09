"""Downstream training for UCR, HAR, TSB-AD, and forecasting."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from .losses import (
    EventSetCriterion,
    adaptive_cardinality_loss,
    anomaly_regularization,
    existence_entropy_loss,
    existence_sparsity_loss,
    event_width_prior_loss,
    program_measurement_consistency_loss,
    task_loss,
)


TASK_ADAPTERS = {
    "morphology": ("UCRAdapter", "UCRDataAdapter"),
    "event_classification": ("HARAdapter", "UCIHARAdapter", "HARDataAdapter"),
    "anomaly": ("TSBADAdapter", "TSBAdAdapter", "AnomalyAdapter"),
    "forecast": ("ForecastAdapter", "ForecastDataAdapter"),
}


@dataclass
class TaskTrainConfig:
    task: str
    dataset: Optional[str] = None
    epochs: int = 20
    freeze_epochs: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    encoder_lr: float = 3e-5
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    checkpoint: Optional[str] = None
    init_checkpoint: Optional[str] = None
    smoke: bool = False
    measurement_consistency_weight: float = 0.1
    cardinality_weight: float = 0.01
    width_prior_weight: float = 0.0
    sparse_weight: float = 0.01
    entropy_weight: float = 0.0
    base_loss_weight: float = 0.5
    damage_weight: float = 0.5
    graph_norm_weight: float = 0.0
    damage_margin: float = 0.0
    # Positive term that rewards the graph path for beating the base path by
    # damage_margin. damage_weight only penalizes the graph being *worse* than
    # base, which lets the residual settle at "no gain"; this gives it an
    # explicit reason to be better.
    graph_advantage_weight: float = 0.0
    anomaly_regularization_weight: float = 0.0
    gradient_clip: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    overfit_samples: int = 0
    max_channels: int = 64
    forecast_loss_mode: str = "mse_only"
    use_class_weights: bool = False
    selection_metric: Optional[str] = None
    data_root: str = "/data/panhongyu/TS-LLM/datasets"
    val_ratio: float = 0.2
    seq_len: int = 96
    pred_len: int = 96
    anomaly_window_size: int = 512
    anomaly_threshold: Optional[float] = None
    anomaly_far_cap: Optional[float] = 0.5
    anomaly_per_series_sweep: bool = False
    balanced_anomaly: bool = False
    disable_graph_correction: bool = False
    ranking_margin: float = 1.0
    detail_log: Optional[str] = None
    detail_samples: int = 2
    detail_max_points: int = 128
    detail_min_epoch: int = 1
    event_aux_weight: float = 0.0
    oracle_events: bool = False
    oracle_train_ratio: float = 1.0
    oracle_curriculum_epochs: int = 0
    oracle_eval_events: bool = True
    # Transfer protocol: freeze event extractor + graph; train task head only.
    freeze_backbone: bool = False
    # Stratified fraction of the TRAIN split only (val/test untouched unless
    # eval_fraction < 1).
    label_fraction: float = 1.0
    # Stratified fraction of VAL/TEST used for per-epoch eval and final test.
    # Independent of label_fraction; use e.g. 0.3 to speed up eval.
    eval_fraction: float = 1.0
    # Force re-init of task_heads after loading init_checkpoint.
    reset_task_head: bool = False

    def __post_init__(self) -> None:
        if self.task not in TASK_ADAPTERS:
            raise ValueError(f"Unsupported downstream task: {self.task}")
        if self.checkpoint is None:
            self.checkpoint = f"checkpoints/{self.task}.pt"
        if self.dataset is None:
            self.dataset = {
                "morphology": "SyntheticControl",
                "event_classification": "UCI_HAR",
                "anomaly": "TSB-AD",
                "forecast": "ETTm1",
            }[self.task]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: Optional[str] = None) -> torch.device:
    if name is None or str(name).strip() == "":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    name = str(name).strip()
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


def build_model(**model_kwargs: Any) -> torch.nn.Module:
    from src.models.grounded_reasoner import GroundedEventGraphReasoner, GroundedReasonerConfig

    config = GroundedReasonerConfig(**model_kwargs)
    return GroundedEventGraphReasoner(config)


def _load_compatible_state(
    model: torch.nn.Module, checkpoint: str
) -> Dict[str, Any]:
    """Load checkpoint weights, keeping a visible audit trail of skips.

    ``strict=False`` loading is intentional (per-dataset task heads change shape),
    but the *reason* for every skipped tensor is now reported: missing keys and
    shape mismatches are logged per tensor, and a ``event_decoder.queries``
    mismatch is flagged as the num_queries inconsistency that silently re-inited
    decoder slots in log/20260904_110944 (stage3/ablation nq=4 evals of nq=6
    checkpoints). Set STRICT_QUERY_INIT=1 to turn that specific mismatch into a
    hard error; leave it 0 when deliberately resizing (Stage1b widens SCM's
    nq=4 decoder to nq=6).
    """
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    source = state.get("model", state)
    current = model.state_dict()
    compatible: Dict[str, Any] = {}
    skipped: Dict[str, Dict[str, Any]] = {}
    for name, value in source.items():
        if name not in current:
            skipped[name] = {
                "reason": "missing_in_model",
                "checkpoint_shape": list(value.shape),
            }
        elif tuple(current[name].shape) != tuple(value.shape):
            skipped[name] = {
                "reason": "shape_mismatch",
                "checkpoint_shape": list(value.shape),
                "model_shape": list(current[name].shape),
            }
        else:
            compatible[name] = value
    model.load_state_dict(compatible, strict=False)

    head_skips = [name for name in skipped if name.startswith("task_heads.")]
    core_skips = [name for name in skipped if not name.startswith("task_heads.")]
    if skipped:
        print(
            json.dumps(
                {
                    "phase": "init_checkpoint",
                    "checkpoint": str(path),
                    "loaded": len(compatible),
                    "skipped_total": len(skipped),
                    "skipped_task_heads": len(head_skips),
                    "skipped_core": [
                        {"name": name, **skipped[name]} for name in core_skips
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for name in core_skips:
        print(
            f"[warn][init_checkpoint] skipped {name}: {skipped[name]}",
            file=sys.stderr,
            flush=True,
        )

    query_skip = skipped.get("event_decoder.queries")
    if query_skip is not None:
        checkpoint_num_queries = int(source["event_decoder.queries"].shape[0])
        model_num_queries = int(current["event_decoder.queries"].shape[0])
        message = (
            "event_decoder.queries init mismatch: checkpoint has "
            f"num_queries={checkpoint_num_queries}, model was built with "
            f"num_queries={model_num_queries}. The query slots are "
            "re-initialized from scratch; valid only for a deliberate resize "
            "(e.g. Stage1b widening SCM's nq=4 decoder to nq=6)."
        )
        print(
            json.dumps(
                {
                    "phase": "init_checkpoint",
                    "warn": "num_queries_mismatch",
                    "checkpoint": str(path),
                    "checkpoint_num_queries": checkpoint_num_queries,
                    "model_num_queries": model_num_queries,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(f"[warn][init_checkpoint] {message}", file=sys.stderr, flush=True)
        if os.environ.get("STRICT_QUERY_INIT", "0") == "1":
            raise ValueError(message)
    return {
        "loaded": len(compatible),
        "skipped": len(skipped),
        "skipped_core": core_skips,
        "checkpoint_num_queries": (
            int(source["event_decoder.queries"].shape[0])
            if "event_decoder.queries" in source
            else None
        ),
    }


def build_task_loaders(
    config: TaskTrainConfig,
) -> Tuple[
    Iterable[Mapping[str, Any]],
    Optional[Iterable[Mapping[str, Any]]],
    Optional[Iterable[Mapping[str, Any]]],
]:
    try:
        data = importlib.import_module("src.data")
    except ImportError as exc:
        raise RuntimeError(
            "Expected src.data with TimeSeriesSample, collate, and task adapters."
        ) from exc
    adapter_class = next(
        (getattr(data, name) for name in TASK_ADAPTERS[config.task] if hasattr(data, name)),
        None,
    )
    if adapter_class is None:
        from torch.utils.data import DataLoader, Subset

        if config.task == "morphology" and hasattr(data, "make_ucr_datasets"):
            datasets = data.make_ucr_datasets(
                name=config.dataset,
                seed=config.seed,
                root=config.data_root,
                val_ratio=config.val_ratio,
            )
        elif config.task == "event_classification":
            if str(config.dataset or "").upper() == "SMD":
                if not hasattr(data, "make_smd_datasets"):
                    raise RuntimeError("src.data needs make_smd_datasets for SMD event classification.")
                datasets = data.make_smd_datasets(
                    seed=config.seed,
                    root=config.data_root,
                    val_ratio=config.val_ratio,
                )
            elif hasattr(data, "make_uea_ucr_datasets") and str(
                config.dataset or ""
            ) in getattr(data, "UEA_UCR_DATASETS", ()):
                # UEA event datasets (classify protocol + classify/models_shape).
                datasets = data.make_uea_ucr_datasets(
                    name=str(config.dataset),
                    seed=config.seed,
                    root=config.data_root,
                    val_ratio=config.val_ratio,
                )
            else:
                datasets = data.make_uci_har_datasets(
                    seed=config.seed, root=Path(config.data_root) / "UCI_HAR"
                )
        elif config.task == "anomaly" and hasattr(data, "make_tsb_ad"):
            if hasattr(data, "split_tsb_ad_tuning_files"):
                train_files, val_files = data.split_tsb_ad_tuning_files(
                    root=config.data_root, seed=config.seed
                )
            else:
                train_files, val_files = None, None
            anomaly_common = {
                "root": config.data_root,
                "window_size": config.anomaly_window_size,
                "max_channels": config.max_channels,
            }
            datasets = {
                "train": data.make_tsb_ad(
                    split="tuning",
                    filenames=train_files,
                    normal_prefix_only=True,
                    include_anomaly=bool(getattr(config, "oracle_events", False)),
                    **anomaly_common,
                ),
                "val": data.make_tsb_ad(
                    split="tuning",
                    filenames=val_files,
                    normal_prefix_only=False,
                    **anomaly_common,
                ),
                "test": data.make_tsb_ad(split="eva", **anomaly_common),
            }
        elif config.task == "forecast" and hasattr(data, "make_forecasting_datasets"):
            datasets = data.make_forecasting_datasets(
                name=config.dataset,
                root=config.data_root,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
            )
        else:
            expected = "/".join(TASK_ADAPTERS[config.task])
            raise RuntimeError(
                f"src.data needs {expected} or a make_* dataset factory for {config.task}."
            )
        loader_kwargs = {
            "batch_size": 1 if config.task == "anomaly" else config.batch_size,
            "collate_fn": data.padding_collate,
        }
        if config.overfit_samples > 0 and config.task in {
            "morphology",
            "forecast",
            "anomaly",
            "event_classification",
        }:
            source = datasets["train"]
            if config.task == "morphology":
                labels = np.asarray(getattr(source, "y"))
                per_class = max(
                    1, config.overfit_samples // max(len(np.unique(labels)), 1)
                )
                indices = np.concatenate(
                    [
                        np.flatnonzero(labels == label)[:per_class]
                        for label in np.unique(labels)
                    ]
                ).tolist()
            else:
                indices = list(range(min(config.overfit_samples, len(source))))
            diagnostic = Subset(source, indices)
            datasets = {"train": diagnostic, "val": diagnostic, "test": diagnostic}
        elif config.task in {"morphology", "event_classification"}:
            label_fraction = float(getattr(config, "label_fraction", 1.0))
            eval_fraction = float(getattr(config, "eval_fraction", 1.0))
            if 0.0 < label_fraction < 1.0:
                # Few-shot transfer: subsample TRAIN only.
                source = datasets["train"]
                labels = np.asarray(getattr(source, "y"))
                indices = _stratified_fraction_indices(
                    labels, label_fraction, config.seed
                )
                datasets = {
                    "train": Subset(source, indices),
                    "val": datasets["val"],
                    "test": datasets["test"],
                }
                print(
                    json.dumps(
                        {
                            "phase": "label_fraction",
                            "fraction": label_fraction,
                            "train_full": int(len(source)),
                            "train_used": int(len(indices)),
                            "seed": int(config.seed),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if 0.0 < eval_fraction < 1.0:
                # Speed knob: stratified subsample of val/test.
                # Use a seed offset so val/test draws differ from train's.
                eval_info = {"phase": "eval_fraction", "fraction": eval_fraction}
                for split, seed_offset in (("val", 1), ("test", 2)):
                    if split not in datasets:
                        continue
                    source = datasets[split]
                    # Subset may wrap an earlier train cut; reach through to .y.
                    base = source
                    while isinstance(base, Subset):
                        base = base.dataset
                    labels = np.asarray(getattr(base, "y"))
                    if isinstance(source, Subset):
                        labels = labels[np.asarray(source.indices)]
                    indices = _stratified_fraction_indices(
                        labels, eval_fraction, int(config.seed) + seed_offset
                    )
                    eval_info[f"{split}_full"] = int(len(source))
                    eval_info[f"{split}_used"] = int(len(indices))
                    datasets[split] = Subset(source, indices)
                eval_info["seed"] = int(config.seed)
                print(json.dumps(eval_info, ensure_ascii=False), flush=True)
        if config.task == "anomaly" and config.balanced_anomaly:
            from torch.utils.data import WeightedRandomSampler

            train_source = datasets["train"]
            weights = np.asarray(train_source.anomaly_weights())
            train_loader = DataLoader(
                train_source,
                batch_size=loader_kwargs["batch_size"],
                collate_fn=loader_kwargs["collate_fn"],
                sampler=WeightedRandomSampler(
                    weights, num_samples=len(weights), replacement=True
                ),
            )
        else:
            train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
        return (
            train_loader,
            DataLoader(datasets.get("val", datasets["test"]), **loader_kwargs),
            DataLoader(datasets["test"], **loader_kwargs)
            if "test" in datasets
            else None,
        )
    kwargs = {"batch_size": config.batch_size, "seed": config.seed, "smoke": config.smoke}
    try:
        adapter = adapter_class(**kwargs)
    except TypeError:
        adapter = adapter_class()
    if hasattr(adapter, "train_dataloader"):
        train_loader = adapter.train_dataloader()
        evaluation_loader = (
            adapter.val_dataloader()
            if hasattr(adapter, "val_dataloader")
            else adapter.test_dataloader()
        )
        test_loader = (
            adapter.test_dataloader() if hasattr(adapter, "test_dataloader") else None
        )
        return train_loader, evaluation_loader, test_loader
    if hasattr(adapter, "build_loader"):
        return (
            adapter.build_loader(split="train"),
            adapter.build_loader(split="val")
            if hasattr(adapter, "build_loader")
            else adapter.build_loader(split="test"),
            adapter.build_loader(split="test"),
        )
    raise RuntimeError(
        f"{adapter_class.__name__} must provide train_dataloader() or build_loader()."
    )


def _set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        for parameter in encoder.parameters():
            parameter.requires_grad = trainable


def _stratified_fraction_indices(
    labels: np.ndarray, fraction: float, seed: int
) -> List[int]:
    """Stratified subsample; at least one example per class when possible."""
    if fraction >= 1.0:
        return list(range(len(labels)))
    if fraction <= 0.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * fraction)))
        n = min(n, len(idx))
        parts.append(idx[:n])
    chosen = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    rng.shuffle(chosen)
    return chosen.astype(int).tolist()


def _reset_task_heads(model: torch.nn.Module) -> None:
    heads = getattr(model, "task_heads", None)
    if heads is None:
        return
    for module in heads.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
        elif isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()


def _configure_ranking_stage(model: torch.nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not enabled:
        return
    compat = getattr(model, "compat_scorer", None)
    if compat is not None:
        for parameter in compat.parameters():
            parameter.requires_grad = True
    gate = getattr(model, "graph_gate", None)
    if isinstance(gate, torch.nn.Parameter):
        gate.requires_grad = True


def _optimizer(model: torch.nn.Module, config: TaskTrainConfig) -> torch.optim.Optimizer:
    encoder, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "encoder" in name else rest).append(parameter)
    groups = []
    if rest:
        groups.append({"params": rest, "lr": config.lr})
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_lr})
    if not groups:
        raise RuntimeError("Model has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _target(batch: Mapping[str, Any]) -> torch.Tensor:
    for key in ("task_target", "task_label", "target", "y", "label", "labels"):
        value = batch.get(key)
        if value is not None:
            if isinstance(value, Mapping) and "point_labels" in value:
                value = value["point_labels"]
                if isinstance(value, list) and len(value) == 1:
                    value = value[0].unsqueeze(0)
            return value
    raise KeyError("Batch needs one of task_target/target/y/label/labels.")


def _forward(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    task: str,
    events_gt: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if "x" not in batch:
        raise KeyError("A collated TimeSeriesSample batch must contain 'x'.")
    kwargs: Dict[str, Any] = {"task": task}
    if events_gt is not None and "events_gt" in inspect.signature(model.forward).parameters:
        kwargs["events_gt"] = events_gt
    output = model(
        batch["x"], batch.get("time_mask"), batch.get("channel_mask"), **kwargs
    )
    if not isinstance(output, Mapping) or "events" not in output:
        raise RuntimeError("Model forward must return a dict containing events.")
    if "task_output" not in output and "outputs" in output:
        task_outputs = output["outputs"]
        if task not in task_outputs:
            raise RuntimeError(f"Model outputs does not contain task {task!r}.")
        output = {**output, "task_output": task_outputs[task]}
    if "task_output" not in output:
        raise RuntimeError("Model forward must return task_output.")
    return output


def _loader_size(loader: Iterable[Mapping[str, Any]]) -> Optional[int]:
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return None


def _progress_interval(total: Optional[int], default: int = 100) -> int:
    if total is None or total <= 0:
        return default
    return max(1, min(default, total // 20 or 1))


def _log_progress(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _batch_sample_ids(batch: Mapping[str, Any], batch_size: int) -> List[str]:
    sample_ids = batch.get("sample_id")
    if sample_ids is None:
        return [f"sample:{index}" for index in range(batch_size)]
    if isinstance(sample_ids, str):
        return [sample_ids]
    return [str(item) for item in sample_ids]


def _selection_score(
    task: str,
    metrics: Mapping[str, float],
    config: Optional[TaskTrainConfig] = None,
) -> tuple[float, str]:
    if config and config.selection_metric:
        metric_name = config.selection_metric
        if metric_name == "mse":
            value = metrics.get("mse", metrics.get("initial/mse", float("inf")))
            return -float(value), metric_name
        value = metrics.get(metric_name, metrics.get(f"initial/{metric_name}"))
        return float(value if value is not None else float("-inf")), metric_name
    if task in {"morphology", "event_classification"}:
        for key in ("macro_f1", "initial/macro_f1", "accuracy", "initial/accuracy"):
            if key in metrics:
                return float(metrics[key]), key.split("/")[-1]
        return float("-inf"), "macro_f1"
    if task == "anomaly":
        # Baseline TSB-AD selects per-file on event-F1; auroc/auc_pr are not
        # primary (auroc is undefined on all-normal files).
        for key in ("event_f1", "auc_pr"):
            if key in metrics:
                return float(metrics[key]), key
        return float("-inf"), "event_f1"
    return -float(metrics.get("mse", float("inf"))), "mse"


def _detail_logger(config: TaskTrainConfig) -> Optional[Any]:
    from src.evaluation.sample_detail_log import SampleDetailLogger

    path = config.detail_log
    if not path:
        logger = SampleDetailLogger.from_env(seed=config.seed)
        return logger
    return SampleDetailLogger(
        path,
        samples_per_epoch=config.detail_samples,
        max_points=config.detail_max_points,
        min_epoch=config.detail_min_epoch,
        seed=config.seed,
    )


@torch.no_grad()
def evaluate_task(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    config: TaskTrainConfig,
    *,
    anomaly_threshold: Optional[float] = None,
    tune_anomaly_threshold: bool = False,
    detail_logger: Optional[Any] = None,
    detail_phase: str = "eval",
    detail_epoch: Optional[int] = None,
    event_aux: Optional[Any] = None,
    oracle_builder: Optional[Any] = None,
) -> Dict[str, float]:
    from src.evaluation import compute_task_metrics
    from src.evaluation.event_metrics import event_metrics as _event_metrics
    from src.evaluation.task_metrics import anomaly_metrics_from_windows

    device = _resolve_device(config.device)
    model.eval()
    if hasattr(model, "_disable_graph_correction"):
        model._disable_graph_correction = config.disable_graph_correction

    targets, predictions = [], []
    anomaly_ids: List[str] = []
    anomaly_labels: List[np.ndarray] = []
    anomaly_scores_init: List[np.ndarray] = []
    event_sums: Dict[str, float] = {}
    event_count = 0

    total = _loader_size(loader)
    interval = _progress_interval(total)
    steps = 0
    detail_logged = 0
    for batch in loader:
        batch = _move(batch, device)
        target = _target(batch)
        events_gt = None
        if (
            getattr(config, "oracle_events", False)
            and getattr(config, "oracle_eval_events", True)
            and oracle_builder is not None
        ):
            events_gt = _batch_oracle_events_collated(batch, oracle_builder, target)
            batch["events_gt"] = events_gt
        model_output = _forward(model, batch, config.task, events_gt=events_gt)
        initial = model_output["task_output"]
        if detail_logger is not None and detail_logged < detail_logger.samples_per_epoch:
            batch_size = batch["x"].shape[0]
            sample_index = detail_logger.rng.randrange(batch_size)
            if (
                event_aux is not None
                and config.task in {"morphology", "event_classification"}
                and "events_gt" not in batch
            ):
                batch["events_gt"] = _batch_events_gt_collated(batch, event_aux)
            detail_logger.log_batch(
                batch,
                model_output,
                task=config.task,
                phase=detail_phase,
                epoch=detail_epoch,
                step=steps + 1,
                sample_indices=[sample_index],
            )
            detail_logged += 1
        if config.task in {"morphology", "event_classification"}:
            initial = initial.argmax(-1)
        elif config.task == "forecast" and target.ndim == 3:
            if initial.shape == target.transpose(1, 2).shape:
                target = target.transpose(1, 2)

        if config.task == "anomaly":
            label_np = target.detach().cpu().numpy()
            init_np = initial.detach().cpu().numpy()
            if label_np.ndim == 1:
                label_np = label_np.reshape(1, -1)
                init_np = init_np.reshape(1, -1)
            for index, sample_id in enumerate(_batch_sample_ids(batch, label_np.shape[0])):
                anomaly_ids.append(sample_id)
                anomaly_labels.append(label_np[index].reshape(-1))
                anomaly_scores_init.append(init_np[index].reshape(-1))
        else:
            targets.append(target.detach().cpu().numpy())
            predictions.append(initial.detach().cpu().numpy())
            if event_aux is not None and config.task == "morphology":
                batch_size = batch["x"].shape[0]
                sample_ids = _batch_sample_ids(batch, batch_size)
                length = batch["x"].shape[-1]
                for batch_index, sample_id in enumerate(sample_ids):
                    parsed = _parse_ucr_sample_id(sample_id)
                    if parsed is None:
                        continue
                    name, split, source = parsed
                    if not event_aux.aligned(name, split) or source >= event_aux.aligned_limit(name, split):
                        continue
                    ground_truth = event_aux.rule_events(name, split, source, length=length)
                    if not ground_truth:
                        continue
                    predicted = _predicted_event_records(model_output["events"], batch_index)
                    metrics = _event_metrics(predicted, ground_truth)
                    for key, value in metrics.items():
                        if np.isfinite(value):
                            event_sums[key] = event_sums.get(key, 0.0) + value
                    event_count += 1

        steps += 1
        if steps == 1 or steps % interval == 0 or (total is not None and steps >= total):
            _log_progress(
                {"task": config.task, "phase": "eval", "step": steps, "steps": total}
            )
        if config.smoke:
            break

    if config.task == "anomaly":
        result = anomaly_metrics_from_windows(
            anomaly_ids,
            anomaly_labels,
            anomaly_scores_init,
            threshold=anomaly_threshold,
            tune_threshold=tune_anomaly_threshold,
            far_cap=config.anomaly_far_cap,
            per_series_sweep=config.anomaly_per_series_sweep,
        )
        _log_progress({"task": config.task, "phase": "eval_metrics", "steps": steps})
        if event_sums and event_count:
            for key, value in event_sums.items():
                result[f"event/{key}"] = value / event_count
        return result

    truth = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    result = compute_task_metrics(config.task, truth, prediction)
    if event_sums and event_count:
        for key, value in event_sums.items():
            result[f"event/{key}"] = value / event_count
    return result


def _scm_events_tensor(events: Any) -> bool:
    return (
        isinstance(events, torch.Tensor)
        and events.ndim == 2
        and events.shape[-1] >= 5
    )


def _events_targets(batch: Mapping[str, Any], model: torch.nn.Module) -> Optional[list]:
    events_gt = batch.get("events_gt")
    if events_gt is None:
        return None
    # Dict format (joint-oracle collation): per-sample lists of tensors.
    if isinstance(events_gt, dict) and "types" in events_gt:
        targets = []
        has_events = False
        n_samples = len(events_gt["types"])
        # Channel width mirrors the event decoder's channel_scores (real
        # channels in this batch), keeping the criterion's BCE aligned.
        channel_mask = batch.get("channel_mask")
        n_channels = (
            int(channel_mask.shape[1])
            if channel_mask is not None
            else model.config.max_channels
        )
        for index in range(n_samples):
            types = events_gt["types"][index]
            if types.numel() == 0:
                targets.append(
                    {
                        "types": torch.zeros(0, dtype=torch.long),
                        "boundaries": torch.zeros(0, 2),
                        "channels": torch.zeros(0, n_channels),
                    }
                )
                continue
            has_events = True
            boundaries = events_gt["boundaries"][index]
            channels = torch.zeros(types.shape[0], n_channels, dtype=torch.float32)
            event_channels = events_gt["event_channels"][index].long().clamp(
                0, n_channels - 1
            )
            channels.scatter_(1, event_channels.unsqueeze(1), 1.0)
            targets.append(
                {
                    "types": types.long(),
                    "boundaries": boundaries,
                    "channels": channels,
                    "attributes": events_gt["attributes"][index],
                    "attribute_mask": events_gt["attribute_mask"][index],
                    "confidence": events_gt["confidence"][index],
                }
            )
        return targets if has_events else None
    # SCM tensor format (Stage-1 grounding pretraining).
    if isinstance(events_gt, list):
        if not any(_scm_events_tensor(events) for events in events_gt):
            return None
    elif not _scm_events_tensor(events_gt):
        return None
    from src.training.grounding_utils import batch_events_targets

    return batch_events_targets(
        events_gt,
        num_types=model.config.num_event_types,
        num_channels=model.config.max_channels,
    )


def _parse_ucr_sample_id(sample_id: str) -> Optional[Tuple[str, str, int]]:
    parts = str(sample_id).split(":")
    if len(parts) >= 4 and parts[0] == "ucr":
        try:
            return parts[1], parts[2], int(parts[3])
        except ValueError:
            return None
    return None


def _predicted_event_records(
    events: Mapping[str, torch.Tensor], batch_index: int
) -> List[Dict[str, Any]]:
    from src.event_schema import PRIMITIVES

    existence = events["existence"][batch_index]
    selected = (existence >= 0.5).nonzero(as_tuple=False).flatten().tolist()
    if not selected:
        return []
    if "start" in events and "end" in events:
        starts, ends = events["start"][batch_index], events["end"][batch_index]
    else:
        starts, ends = events["boundaries"][batch_index, :, 0], events["boundaries"][batch_index, :, 1]
    types = events["type"].argmax(-1)[batch_index]
    records = []
    for slot in selected:
        records.append(
            {
                "type": PRIMITIVES[int(types[slot].item()) % len(PRIMITIVES)],
                "boundary": [float(starts[slot].item()), float(ends[slot].item())],
            }
        )
    return records


def _empty_event_target() -> Dict[str, torch.Tensor]:
    return {
        "types": torch.zeros(0, dtype=torch.long),
        "boundaries": torch.zeros(0, 2, dtype=torch.float32),
        "channels": torch.zeros(0, 1, dtype=torch.float32),
        "attributes": torch.zeros(0, 7, dtype=torch.float32),
        "attribute_mask": torch.zeros(0, 7, dtype=torch.float32),
        "confidence": torch.zeros(0, dtype=torch.float32),
    }


def _batch_events_gt_collated(
    batch: Mapping[str, Any], event_aux: Any
) -> Dict[str, List[torch.Tensor]]:
    """Collated per-sample rule event targets for detail logging."""
    # Online classify aux reuses the oracle builder API (HAR / SMD / UEA).
    if hasattr(event_aux, "event_target_for_sample"):
        return _batch_oracle_events_collated(batch, event_aux, _target(batch))
    sample_ids = _batch_sample_ids(batch, batch["x"].shape[0])
    length = batch["x"].shape[-1]
    types, boundaries, channels, event_channels = [], [], [], []
    for sample_id in sample_ids:
        target = None
        parsed = _parse_ucr_sample_id(sample_id)
        if parsed is not None:
            name, split, source = parsed
            if event_aux.aligned(name, split) and source < event_aux.aligned_limit(
                name, split
            ):
                target = event_aux.event_target(
                    name, split, source, length=length, n_channels=1
                )
        if target is None or target["types"].numel() == 0:
            types.append(torch.zeros(0, dtype=torch.long))
            boundaries.append(torch.zeros(0, 2, dtype=torch.float32))
            channels.append(torch.zeros(0, 1, dtype=torch.float32))
            event_channels.append(torch.zeros(0, dtype=torch.long))
        else:
            types.append(target["types"])
            boundaries.append(target["boundaries"])
            channels.append(target["channels"])
            event_channels.append(target["event_channels"])
    return {
        "types": types,
        "boundaries": boundaries,
        "channels": channels,
        "event_channels": event_channels,
    }


def _batch_oracle_events_collated(
    batch: Mapping[str, Any],
    oracle_builder: Any,
    targets: torch.Tensor,
) -> Dict[str, List[torch.Tensor]]:
    """Collated per-sample oracle GT event targets for joint-oracle training/eval.

    Dispatches by sample_id family (ucr / uci_har / tsb_ad / forecast) through
    :meth:`OracleEventBuilder.event_target_for_sample`.
    """
    sample_ids = _batch_sample_ids(batch, batch["x"].shape[0])
    target_np = targets.detach().cpu().numpy().reshape(-1)
    channel_mask = batch.get("channel_mask")
    types, boundaries, channels, event_channels = [], [], [], []
    attributes, attribute_masks, confidences = [], [], []
    for batch_index, sample_id in enumerate(sample_ids):
        target = None
        x_np = batch["x"][batch_index].detach().cpu().numpy()
        class_id = None
        if batch_index < target_np.size:
            try:
                class_id = int(target_np[batch_index])
            except (TypeError, ValueError):
                class_id = None
        mask_np = None
        if channel_mask is not None:
            mask_np = channel_mask[batch_index].detach().cpu().numpy()
        if hasattr(oracle_builder, "event_target_for_sample"):
            target = oracle_builder.event_target_for_sample(
                sample_id,
                x_np,
                class_id=class_id,
                channel_mask=mask_np,
            )
        else:
            parsed = _parse_ucr_sample_id(sample_id)
            if parsed is not None:
                name, split, source = parsed
                target = oracle_builder.event_target(
                    name,
                    split,
                    source,
                    length=x_np.shape[-1],
                    n_channels=1,
                    raw_x=x_np[0] if x_np.ndim > 1 else x_np,
                    class_id=class_id,
                )
        if target is None or target["types"].numel() == 0:
            types.append(torch.zeros(0, dtype=torch.long))
            boundaries.append(torch.zeros(0, 2, dtype=torch.float32))
            channels.append(torch.zeros(0, 1, dtype=torch.float32))
            event_channels.append(torch.zeros(0, dtype=torch.long))
            attributes.append(torch.zeros(0, 7, dtype=torch.float32))
            attribute_masks.append(torch.zeros(0, 7, dtype=torch.float32))
            confidences.append(torch.zeros(0, dtype=torch.float32))
        else:
            types.append(target["types"])
            boundaries.append(target["boundaries"])
            channels.append(target["channels"])
            event_channels.append(target["event_channels"])
            attributes.append(target["attributes"])
            attribute_masks.append(target["attribute_mask"])
            confidences.append(target["confidence"])
    return {
        "types": types,
        "boundaries": boundaries,
        "channels": channels,
        "event_channels": event_channels,
        "attributes": attributes,
        "attribute_mask": attribute_masks,
        "confidence": confidences,
    }


def _event_aux_targets(
    batch: Mapping[str, Any], event_aux: Any
) -> Optional[List[Dict[str, torch.Tensor]]]:
    """Per-sample rule event targets for the current batch (None when no GT)."""
    sample_ids = _batch_sample_ids(batch, batch["x"].shape[0])
    targets: List[Dict[str, torch.Tensor]] = []
    has_events = False

    # Online classify aux (HAR / SMD / UEA): same labels as oracle builder.
    if hasattr(event_aux, "event_target_for_sample"):
        class_targets = _target(batch).detach().cpu().numpy().reshape(-1)
        channel_mask = batch.get("channel_mask")
        for batch_index, sample_id in enumerate(sample_ids):
            x_np = batch["x"][batch_index].detach().cpu().numpy()
            class_id = None
            if batch_index < class_targets.size:
                try:
                    class_id = int(class_targets[batch_index])
                except (TypeError, ValueError):
                    class_id = None
            mask_np = None
            if channel_mask is not None:
                mask_np = channel_mask[batch_index].detach().cpu().numpy()
            target = event_aux.event_target_for_sample(
                sample_id,
                x_np,
                class_id=class_id,
                channel_mask=mask_np,
            )
            if target is not None and target["types"].numel() > 0:
                has_events = True
                targets.append(target)
            else:
                targets.append(_empty_event_target())
        return targets if has_events else None

    length = batch["x"].shape[-1]
    for sample_id in sample_ids:
        target = None
        parsed = _parse_ucr_sample_id(sample_id)
        if parsed is not None:
            name, split, source = parsed
            if event_aux.aligned(name, split) and source < event_aux.aligned_limit(
                name, split
            ):
                target = event_aux.event_target(
                    name, split, source, length=length, n_channels=1
                )
        if target is not None and target["types"].numel() > 0:
            has_events = True
            targets.append(target)
        else:
            targets.append(_empty_event_target())
    return targets if has_events else None


def _official_ucr_arrays(
    config: TaskTrainConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Official TRAIN/TEST raw matrices for alignment verification."""
    from src.data.ucr import _read_official_file

    path = Path(config.data_root) / config.dataset
    train_file = path / f"{config.dataset}_TRAIN.txt"
    test_file = path / f"{config.dataset}_TEST.txt"
    train_x = None
    test_x = None
    if train_file.is_file():
        try:
            train_x = _read_official_file(train_file)[0]
        except Exception:
            train_x = None
    if test_file.is_file():
        try:
            test_x = _read_official_file(test_file)[0]
        except Exception:
            test_x = None
    return train_x, test_x


def _build_event_aux_index(
    config: TaskTrainConfig,
) -> Tuple[Optional[Any], bool]:
    """Build event auxiliary supervision for Stage-2 / train ablations.

    Returns (index_or_builder, enabled).

    * morphology: MorphologyEventIndex (aligned Stage-1b / rule GT).
    * event_classification: online OracleEventBuilder using classify-baseline
      labels (HAR / SMD / UEA), same source as oracle_events injection.
    """
    if config.event_aux_weight <= 0:
        return None, False
    if config.task == "event_classification":
        from src.data.oracle_events import OracleEventBuilder

        builder = OracleEventBuilder(
            max_events=6,
            data_root=config.data_root,
        )
        print(
            json.dumps(
                {
                    "task": config.task,
                    "phase": "event_aux",
                    "dataset": config.dataset,
                    "enabled": True,
                    "weight": config.event_aux_weight,
                    "mode": "baseline_classify_rules",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return builder, True
    if config.task != "morphology":
        return None, False
    from src.data.morphology_events import MorphologyEventIndex

    index = MorphologyEventIndex(max_events=6)
    train_x, test_x = _official_ucr_arrays(config)
    aligned_splits = 0
    if train_x is not None:
        index.verify_alignment(config.dataset, train_x, "train")
        # UCR validation is a stratified subset of official TRAIN, so the
        # same index mapping covers both "train" and "val" splits.
        index.verify_alignment(config.dataset, train_x, "val")
        if index.aligned(config.dataset, "train"):
            aligned_splits += 1
    if test_x is not None and index.verify_alignment(config.dataset, test_x, "test"):
        aligned_splits += 1
    if aligned_splits == 0 or not (index.aligned(config.dataset, "train") or index.aligned(config.dataset, "test")):
        print(
            json.dumps(
                {
                    "task": config.task,
                    "phase": "event_aux",
                    "dataset": config.dataset,
                    "enabled": False,
                    "reason": (
                        "generated features are not index-aligned with the "
                        "official split; skipping event auxiliary supervision"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return None, False
    print(
        json.dumps(
            {
                "task": config.task,
                "phase": "event_aux",
                "dataset": config.dataset,
                "enabled": True,
                "weight": config.event_aux_weight,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return index, True


def train_task(
    config: TaskTrainConfig,
    model: Optional[torch.nn.Module] = None,
    loaders: Optional[Tuple[Any, ...]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_everything(config.seed)
    device = _resolve_device(config.device)
    model = model or build_model(**(model_kwargs or {}))
    if hasattr(model, "_disable_graph_correction"):
        model._disable_graph_correction = config.disable_graph_correction

    load_report: Optional[Dict[str, Any]] = None
    if config.init_checkpoint:
        load_report = _load_compatible_state(model, config.init_checkpoint)
    if getattr(config, "reset_task_head", False):
        _reset_task_heads(model)
        print(
            json.dumps({"phase": "reset_task_head", "enabled": True}, ensure_ascii=False),
            flush=True,
        )
    model.to(device)

    resolved_loaders = loaders or build_task_loaders(config)
    if len(resolved_loaders) == 2:
        train_loader, evaluation_loader = resolved_loaders
        test_loader = None
    elif len(resolved_loaders) == 3:
        train_loader, evaluation_loader, test_loader = resolved_loaders
    else:
        raise ValueError("loaders must contain train/val or train/val/test")
    event_aux, event_aux_enabled = _build_event_aux_index(config)

    oracle_builder = None
    if getattr(config, "oracle_events", False):
        from src.data.oracle_events import OracleEventBuilder

        oracle_builder = OracleEventBuilder(
            max_events=getattr(model.config, "num_queries", 6),
            data_root=config.data_root,
        )
        aligned_train = aligned_test = False
        if config.task == "morphology":
            train_x, test_x = _official_ucr_arrays(config)
            if train_x is not None:
                oracle_builder.verify(config.dataset, train_x, "train")
                oracle_builder.verify(config.dataset, train_x, "val")
            if test_x is not None:
                oracle_builder.verify(config.dataset, test_x, "test")
            aligned_train = oracle_builder.aligned(config.dataset, "train")
            aligned_test = oracle_builder.aligned(config.dataset, "test")
        print(
            json.dumps(
                {
                    "task": config.task,
                    "phase": "oracle_events",
                    "dataset": config.dataset,
                    "enabled": True,
                    "aligned_train": aligned_train,
                    "aligned_test": aligned_test,
                    "mode": (
                        "ucr_aligned_or_rule"
                        if config.task == "morphology"
                        else "baseline_classify_rules"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ranking_stage = (
        config.measurement_consistency_weight <= 0 and config.graph_norm_weight > 0
    )
    if ranking_stage and config.epochs > 0:
        _configure_ranking_stage(model, True)

    train_steps = _loader_size(train_loader)
    eval_steps = _loader_size(evaluation_loader) if evaluation_loader is not None else 0
    test_steps = _loader_size(test_loader) if test_loader is not None else 0
    detail_logger = _detail_logger(config)
    epochs = 1 if config.smoke else config.epochs
    _log_progress(
        {
            "task": config.task,
            "phase": "start",
            "train_steps": train_steps,
            "eval_steps": eval_steps,
            "test_steps": test_steps,
            "batch_size": 1 if config.task == "anomaly" else config.batch_size,
            "epochs": 0 if config.epochs == 0 else epochs,
            "ranking_stage": ranking_stage,
            "detail_log": str(detail_logger.path) if detail_logger else None,
        }
    )

    if config.epochs == 0:
        val_metrics = (
            evaluate_task(
                model,
                evaluation_loader,
                config,
                detail_logger=detail_logger,
                detail_phase="eval",
                event_aux=event_aux if event_aux_enabled else None,
                oracle_builder=oracle_builder,
            )
            if evaluation_loader
            else {}
        )
        test_metrics = (
            evaluate_task(
                model,
                test_loader,
                config,
                detail_logger=detail_logger,
                detail_phase="test",
                event_aux=event_aux if event_aux_enabled else None,
                oracle_builder=oracle_builder,
            )
            if test_loader is not None
            else val_metrics
        )
        return {
            "task": config.task,
            "dataset": config.dataset,
            "epochs": 0,
            "metrics": test_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint": config.checkpoint,
            "device": str(device),
            "init_checkpoint": load_report,
            "detail_log": str(detail_logger.path) if detail_logger else None,
        }

    planned_detail_steps = (
        detail_logger.planned_train_steps(train_steps or 0, epochs)
        if detail_logger
        else {}
    )
    if getattr(config, "freeze_backbone", False):
        from .stages import freeze_backbone_train_heads

        freeze_backbone_train_heads(model, task=config.task)
        # Optionally keep graph residual in forward; caller can still pass
        # --disable-graph-correction. Zero event-shaping losses under freeze so
        # frozen decoder slots are not pushed by unused aux terms.
        print(
            json.dumps(
                {
                    "phase": "freeze_backbone",
                    "task": config.task,
                    "trainable": [
                        name
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        _set_encoder_trainable(model, config.freeze_epochs <= 0)
    optimizer = _optimizer(model, config)
    event_criterion = EventSetCriterion()
    history: list[float] = []
    epoch_records: list[dict] = []
    progress_every = _progress_interval(train_steps)
    path = Path(config.checkpoint or f"checkpoints/{config.task}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0
    rng = random.Random(config.seed)

    for epoch in range(epochs):
        if (
            not getattr(config, "freeze_backbone", False)
            and epoch == config.freeze_epochs
            and config.freeze_epochs > 0
            and not ranking_stage
        ):
            _set_encoder_trainable(model, True)
            optimizer = _optimizer(model, config)
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            batch = _move(batch, device)
            target = _target(batch)
            events_gt = None
            if (
                getattr(config, "oracle_events", False)
                and oracle_builder is not None
            ):
                ratio = float(getattr(config, "oracle_train_ratio", 1.0))
                curriculum = int(getattr(config, "oracle_curriculum_epochs", 0))
                if curriculum > 0 and (epoch + 1) <= curriculum:
                    ratio = 1.0
                if ratio >= 1.0 or (ratio > 0 and rng.random() < ratio):
                    events_gt = _batch_oracle_events_collated(batch, oracle_builder, target)
                    batch["events_gt"] = events_gt
            output = _forward(model, batch, config.task, events_gt=events_gt)
            final_loss = task_loss(output["task_output"], target, config.task)
            loss = final_loss
            base_prediction = output.get("outputs", {}).get(f"base/{config.task}")
            base_loss = None
            damage = None
            if base_prediction is not None:
                base_loss = task_loss(base_prediction, target, config.task)
                loss = loss + config.base_loss_weight * base_loss
                damage = torch.relu(
                    final_loss - base_loss.detach() - config.damage_margin
                )
                loss = loss + config.damage_weight * damage
                if config.graph_advantage_weight > 0:
                    # Maximize (base - graph) once it clears the margin, so the
                    # graph residual is pushed to actually improve on the base path
                    # instead of merely not hurting it.
                    advantage = torch.relu(
                        base_loss.detach() - final_loss - config.damage_margin
                    )
                    loss = loss - config.graph_advantage_weight * advantage
            if config.cardinality_weight > 0:
                loss = loss + config.cardinality_weight * adaptive_cardinality_loss(
                    output["events"], batch["x"], config.task
                )
            if config.sparse_weight > 0:
                loss = loss + config.sparse_weight * existence_sparsity_loss(
                    output["events"]["existence"]
                )
            if config.entropy_weight > 0:
                loss = loss + config.entropy_weight * existence_entropy_loss(
                    output["events"]["existence"]
                )
            if config.width_prior_weight > 0:
                loss = loss + config.width_prior_weight * event_width_prior_loss(
                    output["events"]
                )
            if config.task == "anomaly" and config.anomaly_regularization_weight > 0:
                loss = loss + config.anomaly_regularization_weight * anomaly_regularization(
                    output["task_output"], target
                )
            event_targets = _events_targets(batch, model)
            if event_targets is not None and not ranking_stage:
                event_loss, _ = event_criterion(output["events"], event_targets)
                loss = loss + 0.1 * event_loss
                # In oracle mode the supervised events are the injected GT, so the
                # decoder itself never sees a gradient toward real detection. Add
                # the same criterion on the decoder's own predictions so that it
                # learns to detect events on the eval (real decoder) path.
                # event_classification (UCI HAR) is always evaluated with injected
                # GT events under joint-oracle, so supervising the raw decoder here
                # adds only gradient noise; it lowers measured accuracy (~0.66 vs
                # ~0.70 with the flag off) and is skipped for that task.
                decoder_events = output.get("decoder_events")
                if (
                    decoder_events is not None
                    and config.task != "event_classification"
                    and config.event_aux_weight > 0
                ):
                    decoder_loss, _ = event_criterion(decoder_events, event_targets)
                    loss = loss + config.event_aux_weight * decoder_loss
            if event_aux_enabled and config.event_aux_weight > 0 and not ranking_stage:
                aux_targets = _event_aux_targets(batch, event_aux)
                if aux_targets is not None:
                    aux_events = output.get("decoder_events")
                    if aux_events is None:
                        aux_events = output["events"]
                    aux_loss, _ = event_criterion(aux_events, aux_targets)
                    loss = loss + config.event_aux_weight * aux_loss
            consistency_weight = config.measurement_consistency_weight * min(
                1.0, max(0.0, (epoch + 1) / max(epochs, 1))
            )
            if consistency_weight > 0:
                consistency = program_measurement_consistency_loss(
                    batch["x"],
                    output["events"],
                    batch["time_mask"],
                    batch["channel_mask"],
                )
                loss = loss + consistency_weight * consistency
            if ranking_stage and batch.get("events_gt") is not None:
                from src.training.grounding_utils import compatibility_loss

                rank_loss, _ = compatibility_loss(
                    model,
                    batch,
                    num_types=model.config.num_event_types,
                    num_channels=model.config.max_channels,
                    rng=rng,
                )
                loss = loss + rank_loss
            if config.graph_norm_weight > 0 and "graph_delta" in output:
                loss = loss + config.graph_norm_weight * output["graph_delta"].pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if (
                detail_logger is not None
                and (epoch + 1) in planned_detail_steps
                and steps in planned_detail_steps[epoch + 1]
            ):
                if (
                    event_aux_enabled
                    and config.task in {"morphology", "event_classification"}
                    and batch.get("events_gt") is None
                ):
                    batch["events_gt"] = _batch_events_gt_collated(batch, event_aux)
                detail_logger.log_batch(
                    batch,
                    output,
                    task=config.task,
                    phase="train",
                    epoch=epoch + 1,
                    step=steps,
                )
            if steps == 1 or steps % progress_every == 0 or (
                train_steps is not None and steps >= train_steps
            ):
                _log_progress(
                    {
                        "task": config.task,
                        "phase": "train",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": steps,
                        "steps": train_steps,
                        "train_loss": running / steps,
                        "base_loss": float(base_loss.detach()) if base_loss is not None else None,
                        "damage_loss": float(damage.detach()) if damage is not None else None,
                        "event_count": float(
                            output["events"]["existence"].detach().sum(dim=-1).mean()
                        ),
                    }
                )
            if config.smoke:
                break

        train_loss = running / max(steps, 1)
        history.append(train_loss)
        metrics = (
            evaluate_task(
                model,
                evaluation_loader,
                config,
                tune_anomaly_threshold=config.task == "anomaly",
                detail_logger=detail_logger,
                detail_phase="val",
                detail_epoch=epoch + 1,
                event_aux=event_aux if event_aux_enabled else None,
                oracle_builder=oracle_builder,
            )
            if evaluation_loader
            else {}
        )
        score, selection_metric = _selection_score(config.task, metrics, config)
        improved = score > best_score + config.min_delta
        if improved:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            stale_epochs = 0
            if config.task == "anomaly":
                config.anomaly_threshold = metrics.get("threshold")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "loss": history,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "selection_metric": selection_metric,
                },
                path,
            )
        else:
            stale_epochs += 1
        record = {
            "task": config.task,
            "epoch": epoch + 1,
            "epochs": epochs,
            "train_loss": train_loss,
            "metrics": metrics,
            "selection_metric": selection_metric,
            "is_best": improved,
            "best_epoch": best_epoch,
        }
        epoch_records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if not config.smoke and config.patience > 0 and stale_epochs >= config.patience:
            _log_progress(
                {
                    "task": config.task,
                    "phase": "early_stop",
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                }
            )
            break

    if best_epoch == 0:
        torch.save({"model": model.state_dict(), "config": asdict(config), "loss": history}, path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = (
        evaluate_task(
            model,
            test_loader,
            config,
            anomaly_threshold=config.anomaly_threshold,
            detail_logger=detail_logger,
            detail_phase="test",
            event_aux=event_aux if event_aux_enabled else None,
            oracle_builder=oracle_builder,
        )
        if test_loader is not None
        else best_metrics
    )
    return {
        "task": config.task,
        "dataset": config.dataset,
        "epochs": epochs,
        "train_loss": history[-1] if history else None,
        "metrics": test_metrics,
        "val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "epoch_history": epoch_records,
        "checkpoint": str(path),
        "device": str(device),
        "init_checkpoint": load_report,
        "detail_log": str(detail_logger.path) if detail_logger else None,
    }
