"""Stage-1 Controlled-SCM event grounding pretraining."""

from __future__ import annotations

import importlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from .losses import (
    EventSetCriterion,
    event_width_prior_loss,
    existence_entropy_loss,
    existence_sparsity_loss,
)


@dataclass
class EventPretrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    checkpoint: str = "checkpoints/controlled_scm_stage1.pt"
    smoke: bool = False
    relation_weight: float = 0.5
    reconstruction_weight: float = 0.5
    sparse_weight: float = 0.01
    entropy_weight: float = 0.0
    width_prior_weight: float = 0.0
    channel_loss_weight: float = 0.5
    detail_log: Optional[str] = None
    detail_samples: int = 2
    detail_max_points: int = 128
    detail_min_epoch: int = 1
    morphology_roots: Optional[Mapping[str, str]] = None
    morphology_datasets: Optional[Sequence[str]] = None
    max_events: int = 6
    use_raw_channel_only: bool = False
    stage_name: str = "controlled_scm_stage1"
    task_name: str = "controlled_scm"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def _normalize_event_targets(
    targets: Any, channel_count: int
) -> List[Dict[str, torch.Tensor]]:
    if targets is None:
        raise KeyError("Batch must contain events/event_targets/events_gt.")
    if isinstance(targets, Mapping):
        values = list(targets.values())
        if not values:
            raise ValueError("Event targets must be non-empty.")
        first = values[0]
        if torch.is_tensor(first) and first.ndim >= 2:
            batch_size = int(first.shape[0])
            return [
                {key: value[batch_index] for key, value in targets.items()}
                for batch_index in range(batch_size)
            ]
        if isinstance(first, (list, tuple)):
            batch_size = len(first)
            return [
                {key: value[batch_index] for key, value in targets.items()}
                for batch_index in range(batch_size)
            ]
        return [dict(targets)]
    if torch.is_tensor(targets):
        targets = [row for row in targets]
    if not targets:
        raise ValueError("Event targets must be non-empty.")
    if isinstance(targets[0], dict):
        return [dict(row) for row in targets]
    converted: List[Dict[str, torch.Tensor]] = []
    for rows in targets:
        channels = torch.zeros(rows.shape[0], channel_count, device=rows.device)
        channels.scatter_(1, rows[:, 3].long().unsqueeze(1), 1.0)
        converted.append(
            {
                "types": rows[:, 0].long(),
                "boundaries": rows[:, 1:3],
                "channels": channels,
                "event_channels": rows[:, 3].long(),
            }
        )
    return converted


def build_morphology_event_loader(config: EventPretrainConfig) -> Iterable[Mapping[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.morphology_events import build_morphology_event_loader as _build_loader

    roots = getattr(config, "morphology_roots", None)
    datasets = getattr(config, "morphology_datasets", None)
    use_raw_channel_only = bool(getattr(config, "use_raw_channel_only", False))
    max_events = int(getattr(config, "max_events", 6))
    loader = _build_loader(
        batch_size=2 if config.smoke else 32,
        smoke=config.smoke,
        max_events=max_events,
        roots=roots,
        datasets=datasets,
        use_raw_channel_only=use_raw_channel_only,
    )
    return loader


def build_controlled_scm_loader(config: EventPretrainConfig) -> Iterable[Mapping[str, Any]]:
    """Resolve a project adapter lazily so this module remains independently importable."""
    try:
        data = importlib.import_module("src.data")
    except ImportError as exc:
        raise RuntimeError(
            "Controlled-SCM requires src.data. Provide a ControlledSCMAdapter "
            "(or build_controlled_scm_loader) yielding x, masks, and event targets."
        ) from exc
    for name in ("ControlledSCMAdapter", "ControlledSCMDataAdapter"):
        adapter_class = getattr(data, name, None)
        if adapter_class is not None:
            adapter = adapter_class(smoke=config.smoke, seed=config.seed)
            if hasattr(adapter, "train_dataloader"):
                return adapter.train_dataloader()
            if hasattr(adapter, "build_loader"):
                return adapter.build_loader(split="train")
    if hasattr(data, "make_controlled_scm") and hasattr(data, "padding_collate"):
        from torch.utils.data import DataLoader

        dataset = data.make_controlled_scm(
            size=8 if config.smoke else 10_000, split="train", seed=config.seed
        )
        return DataLoader(
            dataset,
            batch_size=2 if config.smoke else 32,
            shuffle=True,
            collate_fn=data.padding_collate,
        )
    raise RuntimeError(
        "src.data has no ControlledSCMAdapter. Expected train_dataloader() or "
        "build_loader(split='train')."
    )


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


def pretrain_events(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    config: Optional[EventPretrainConfig] = None,
) -> Dict[str, Any]:
    config = config or EventPretrainConfig()
    seed_everything(config.seed)
    device = _device(config.device)
    model.to(device).train()
    criterion = EventSetCriterion(
        weights={
            "exist": 1.0,
            "type": 1.0,
            "boundary": 2.0,
            "iou": 2.0,
            "attribute": 0.5,
            "channel": config.channel_loss_weight,
            "confidence": 0.5,
        }
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    history = []
    epochs = 1 if config.smoke else config.epochs
    train_steps = _loader_size(loader)
    progress_every = _progress_interval(train_steps)
    from src.evaluation.sample_detail_log import SampleDetailLogger

    detail_logger: Optional[SampleDetailLogger]
    if config.detail_log:
        detail_logger = SampleDetailLogger(
            config.detail_log,
            samples_per_epoch=config.detail_samples,
            max_points=config.detail_max_points,
            min_epoch=config.detail_min_epoch,
            seed=config.seed,
        )
    else:
        detail_logger = SampleDetailLogger.from_env(seed=config.seed)
    planned_detail_steps = (
        detail_logger.planned_train_steps(train_steps or 0, epochs)
        if detail_logger
        else {}
    )
    _log_progress(
        {
            "task": config.task_name,
            "phase": "start",
            "train_steps": train_steps,
            "epochs": epochs,
            "device": str(device),
            "detail_log": str(detail_logger.path) if detail_logger else None,
        }
    )
    for epoch in range(epochs):
        running, steps = 0.0, 0
        for batch in loader:
            batch = _move(batch, device)
            try:
                output = model(
                    batch["x"],
                    batch.get("time_mask"),
                    batch.get("channel_mask"),
                    task=config.task_name,
                )
            except ValueError as exc:
                if "unsupported task" not in str(exc):
                    raise
                output = model(
                    batch["x"],
                    batch.get("time_mask"),
                    batch.get("channel_mask"),
                    task="all",
                )
            if "events" not in output:
                raise KeyError("Model output must contain an 'events' dictionary.")
            targets = batch.get(
                "events", batch.get("event_targets", batch.get("events_gt"))
            )
            channel_values = output["events"].get(
                "channel_logits", output["events"].get("channel_scores")
            )
            targets = _normalize_event_targets(targets, channel_values.shape[-1])
            event_loss, _ = criterion(output["events"], targets)
            reconstruction = output.get("reconstruction")
            if reconstruction is None:
                reconstruction_loss = event_loss.new_zeros(())
            else:
                valid = (
                    batch["time_mask"][:, None, :].to(reconstruction.dtype)
                    * batch["channel_mask"][:, :, None].to(reconstruction.dtype)
                )
                reconstruction_loss = (
                    ((reconstruction - batch["x"]).square() * valid).sum()
                    / valid.sum().clamp_min(1.0)
                )

            relation_loss = event_loss.new_zeros(())
            relation_rows = batch.get("relations_gt")
            graph = output.get("graph")
            if relation_rows is not None and graph is not None:
                matches = criterion.matcher(output["events"], targets)
                relation_terms = []
                for batch_index, (source_slots, target_events) in enumerate(matches):
                    rows = (
                        relation_rows[batch_index]
                        if isinstance(relation_rows, list)
                        else relation_rows[batch_index]
                    )
                    if source_slots.numel() < 2:
                        continue
                    channel_edges = {
                        (int(row[0]), int(row[1]))
                        for row in rows.detach().cpu()
                    }
                    event_channels = targets[batch_index]["event_channels"]
                    for left in range(source_slots.numel()):
                        for right in range(source_slots.numel()):
                            if left == right:
                                continue
                            src_slot = source_slots[left]
                            dst_slot = source_slots[right]
                            src_event = target_events[left]
                            dst_event = target_events[right]
                            target_value = float(
                                (
                                    int(event_channels[src_event]),
                                    int(event_channels[dst_event]),
                                )
                                in channel_edges
                            )
                            score = graph["relation_scores"][
                                batch_index, src_slot, dst_slot, 7
                            ].clamp(1e-6, 1.0 - 1e-6)
                            relation_terms.append(
                                torch.nn.functional.binary_cross_entropy(
                                    score, score.new_tensor(target_value)
                                )
                            )
                if relation_terms:
                    relation_loss = torch.stack(relation_terms).mean()
            loss = (
                event_loss
                + config.relation_weight * relation_loss
                + config.reconstruction_weight * reconstruction_loss
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if (
                detail_logger is not None
                and (epoch + 1) in planned_detail_steps
                and steps in planned_detail_steps[epoch + 1]
            ):
                detail_logger.log_batch(
                    batch,
                    output,
                    task=config.task_name,
                    phase="train",
                    epoch=epoch + 1,
                    step=steps,
                )
            if steps == 1 or steps % progress_every == 0 or (
                train_steps is not None and steps >= train_steps
            ):
                _log_progress(
                    {
                        "task": config.task_name,
                        "phase": "train",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": steps,
                        "steps": train_steps,
                        "train_loss": running / steps,
                    }
                )
            if config.smoke:
                break
        epoch_loss = running / max(steps, 1)
        history.append(epoch_loss)
        _log_progress(
            {
                "task": config.task_name,
                "stage": config.stage_name,
                "phase": "epoch",
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_loss": epoch_loss,
            }
        )
    path = Path(config.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": asdict(config), "loss": history}, path
    )
    return {
        "stage": config.stage_name,
        "epochs": epochs,
        "loss": history[-1] if history else None,
        "epoch_losses": history,
        "checkpoint": str(path),
        "device": str(device),
        "detail_log": str(detail_logger.path) if detail_logger else None,
    }
