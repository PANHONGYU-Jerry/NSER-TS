"""Stage 1: temporal-graph grounding pretraining."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from src.data.grounding_mix import GroundingMixConfig, build_grounding_loader
from src.training.grounding_utils import (
    batch_events_targets,
    compatibility_loss,
    compatibility_metrics,
    relation_supervision_loss,
)
from src.training.losses import EventSetCriterion


@dataclass
class GroundingPretrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    checkpoint: str = "checkpoints/stage1_grounding.pt"
    smoke: bool = False
    extract_weight: float = 1.0
    relation_weight: float = 0.0
    compat_weight: float = 1.0
    compat_temperature: float = 1.0
    seq_len: int = 128
    pred_len: int = 16
    max_channels: int = 64
    data_root: str = "/data/panhongyu/TS-LLM/datasets"
    scm_fraction: float = 0.4
    injection_fraction: float = 0.4
    raw_fraction: float = 0.2
    min_compat_auroc: float = 0.75
    min_compat_ranking_acc: float = 0.70
    min_primitive_macro_f1: float = 0.35
    min_interval_miou: float = 0.45


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


def _configure_stage1(model: torch.nn.Module) -> None:
    """Freeze V0 encoder/decoders; train event graph modules only."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in (
        "event_decoder",
        "event_field_projection",
        "global_event_position",
        "relative_point_position",
        "graph_reasoner",
        "graph_to_temporal",
        "compat_scorer",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    model._training_stage = "1"
    model._disable_graph_correction = True


def build_grounding_pretrain_loader(config: GroundingPretrainConfig) -> DataLoader:
    mix = GroundingMixConfig(
        scm_fraction=config.scm_fraction,
        injection_fraction=config.injection_fraction,
        raw_fraction=config.raw_fraction,
        scm_size=400 if config.smoke else 4000,
        injection_size=400 if config.smoke else 4000,
        raw_size=200 if config.smoke else 2000,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        max_channels=config.max_channels,
        data_root=config.data_root,
        seed=config.seed,
        smoke=config.smoke,
    )
    return build_grounding_loader(mix, batch_size=2 if config.smoke else 16)


def _extract_metrics(
    events_output: Mapping[str, torch.Tensor],
    targets: list[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Lightweight primitive / interval metrics for Stage 1 gates."""

    from src.training.losses import EventSetCriterion

    criterion = EventSetCriterion()
    with torch.no_grad():
        loss, details = criterion(events_output, targets)
    type_acc = []
    ious = []
    for batch_index, target in enumerate(targets):
        if target["types"].numel() == 0:
            continue
        pred_types = events_output["type"][batch_index].argmax(dim=-1)
        pred_exist = events_output["existence"][batch_index] >= 0.5
        gt_types = target["types"]
        matched = min(pred_types[pred_exist].numel(), gt_types.numel())
        if matched > 0:
            type_acc.append(
                float((pred_types[:matched] == gt_types[:matched]).float().mean())
            )
        pred_start = events_output["start"][batch_index, : matched]
        pred_end = events_output["end"][batch_index, :matched]
        gt_start = target["boundaries"][:matched, 0]
        gt_end = target["boundaries"][:matched, 1]
        inter = (torch.minimum(pred_end, gt_end) - torch.maximum(pred_start, gt_start)).clamp_min(0.0)
        union = (torch.maximum(pred_end, gt_end) - torch.minimum(pred_start, gt_start)).clamp_min(1e-4)
        ious.append(float((inter / union).mean()))
    return {
        "extract_loss": float(loss.detach()),
        "primitive_macro_f1": float(sum(type_acc) / len(type_acc)) if type_acc else 0.0,
        "interval_miou": float(sum(ious) / len(ious)) if ious else 0.0,
        **{key: float(value) for key, value in details.items() if torch.is_tensor(value)},
    }


def _stage1_passes_gates(metrics: Mapping[str, float], config: GroundingPretrainConfig) -> bool:
    return (
        float(metrics.get("compat_auroc", 0.0)) >= config.min_compat_auroc
        and float(metrics.get("compat_ranking_acc", 0.0)) >= config.min_compat_ranking_acc
        and float(metrics.get("primitive_macro_f1", 0.0)) >= config.min_primitive_macro_f1
        and float(metrics.get("interval_miou", 0.0)) >= config.min_interval_miou
    )


def pretrain_grounding(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    config: Optional[GroundingPretrainConfig] = None,
) -> Dict[str, Any]:
    config = config or GroundingPretrainConfig()
    seed_everything(config.seed)
    device = _device(config.device)
    model.to(device)

    criterion = EventSetCriterion()
    rng = random.Random(config.seed)
    history = []
    epochs = 1 if config.smoke else config.epochs
    num_types = getattr(model.config, "num_event_types", 9)
    num_channels = getattr(model.config, "max_channels", config.max_channels)
    gate_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        _configure_stage1(model)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=config.lr, weight_decay=config.weight_decay
        )

        model.train()
        totals = {
            "total": 0.0,
            "extract": 0.0,
            "relation": 0.0,
            "compat": 0.0,
        }
        compat_diag = {
            "positive_score_mean": 0.0,
            "negative_score_mean": 0.0,
            "positive_negative_margin": 0.0,
        }
        steps = 0
        for batch in loader:
            batch = _move(batch, device)
            channel_dim = batch["channel_mask"].shape[1]
            events_batches = batch.get("events_list") or batch.get("events_gt") or []
            has_graph = any(
                events is not None and torch.is_tensor(events) and events.numel() > 0
                for events in events_batches
            )
            loss = batch["x"].sum() * 0.0
            extract_loss = loss
            relation_loss = loss
            compat = loss

            if has_graph:
                output = model(
                    batch["x"],
                    batch["time_mask"],
                    batch["channel_mask"],
                    task="all",
                    time_features=batch.get("time_features"),
                )
                targets = batch_events_targets(
                    events_batches,
                    num_types=num_types,
                    num_channels=channel_dim,
                )
                extract_loss, _ = criterion(output["events"], targets)
                existence = output["events"]["existence"] >= model.graph_builder.existence_threshold
                relation_loss = relation_supervision_loss(
                    output["graph"]["relation_scores"],
                    output["graph"]["relation_scores"].detach(),
                    output["graph"]["edge_mask"],
                    existence_mask=existence,
                )
                compat, step_diag = compatibility_loss(
                    model,
                    batch,
                    num_types=num_types,
                    num_channels=channel_dim,
                    temperature=config.compat_temperature,
                    rng=rng,
                )
                for key, value in step_diag.items():
                    compat_diag[key] = compat_diag.get(key, 0.0) + value
                loss = (
                    config.extract_weight * extract_loss
                    + config.relation_weight * relation_loss
                    + config.compat_weight * compat
                )

            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                )
            else:
                grad_norm = 0.0
            optimizer.step()
            steps += 1
            totals["total"] += float(loss.detach())
            totals["extract"] += float(extract_loss.detach())
            totals["relation"] += float(relation_loss.detach())
            totals["compat"] += float(compat.detach())
            if config.smoke:
                break

        compat_diag = {key: value / max(steps, 1) for key, value in compat_diag.items()}
        eval_metrics: Dict[str, float] = {}
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move(batch, device)
                output = model(
                    batch["x"],
                    batch["time_mask"],
                    batch["channel_mask"],
                    task="all",
                    time_features=batch.get("time_features"),
                )
                eval_events = batch.get("events_list") or batch.get("events_gt") or []
                targets = batch_events_targets(
                    eval_events,
                    num_types=num_types,
                    num_channels=batch["channel_mask"].shape[1],
                )
                eval_metrics.update(_extract_metrics(output["events"], targets))
                eval_metrics.update(
                    compatibility_metrics(
                        model,
                        batch,
                        num_types=num_types,
                        num_channels=batch["channel_mask"].shape[1],
                        rng=rng,
                    )
                )
                break

        record = {key: value / max(steps, 1) for key, value in totals.items()}
        record.update(compat_diag)
        record.update(eval_metrics)
        record.update(
            {
                "epoch": epoch + 1,
                "compat_grad_norm": grad_norm,
                "stage1_gate_pass": _stage1_passes_gates(record, config),
            }
        )
        gate_metrics = record
        history.append(record)
        print(
            json.dumps({"task": "stage1_grounding", **record}, ensure_ascii=False),
            flush=True,
        )

    path = Path(config.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "history": history,
            "stage": "1",
            "stage1_gate_pass": bool(gate_metrics.get("stage1_gate_pass", False)),
        },
        path,
    )
    return {
        "stage": "1",
        "checkpoint": str(path),
        "history": history,
        "loss": history[-1]["total"] if history else None,
        "stage1_gate_pass": bool(gate_metrics.get("stage1_gate_pass", False)),
    }
