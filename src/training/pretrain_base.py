"""Stage 0: base temporal pretraining on raw real series."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.common import padding_collate
from src.data.real_background_injection import RealBackgroundInjectionDataset


@dataclass
class BasePretrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    encoder_lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    checkpoint: str = "checkpoints/stage0_base.pt"
    smoke: bool = False
    seq_len: int = 128
    pred_len: int = 16
    max_channels: int = 64
    data_root: str = "/data/panhongyu/TS-LLM/datasets"
    mask_ratio: float = 0.15
    recon_weight: float = 1.0
    forecast_weight: float = 0.5


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


def _configure_stage0(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("encoder", "reconstruction_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    task_heads = getattr(model, "task_heads", None)
    if task_heads is not None and hasattr(task_heads, "forecast_expert"):
        for parameter in task_heads.forecast_expert.data_driven.parameters():
            parameter.requires_grad = True
    model._training_stage = "0"
    model._disable_graph_correction = True


def _sample_masks(time_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    masks = []
    for row in time_mask:
        valid = int(row.sum().item())
        count = max(1, int(round(valid * ratio)))
        mask = torch.zeros_like(row)
        start = torch.randint(0, max(1, valid - count + 1), (1,)).item()
        mask[start : start + count] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


def _base_losses(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    config: BasePretrainConfig,
) -> Dict[str, torch.Tensor]:
    x = batch["x"]
    time_mask = batch["time_mask"]
    channel_mask = batch["channel_mask"]
    mask = _sample_masks(time_mask, config.mask_ratio)
    masked_x = x * (~mask[:, None, :]).to(x.dtype)
    encoded, normalization = model.encoder(
        masked_x, time_mask, channel_mask, time_features=batch.get("time_features")
    )
    channels = channel_mask.shape[1]
    recon_norm = model.reconstruction_head(encoded)[..., :channels].transpose(1, 2)
    recon = recon_norm * normalization["scale"] + normalization["mean"]
    valid = (
        time_mask[:, None, :].to(recon.dtype)
        * channel_mask[:, :, None].to(recon.dtype)
        * mask[:, None, :].to(recon.dtype)
    )
    recon_loss = ((recon - x).square() * valid).sum() / valid.sum().clamp_min(1.0)

    pred_len = min(config.pred_len, x.shape[-1] // 4)
    if pred_len <= 1 or not hasattr(model.task_heads, "forecast_expert"):
        return {"recon": recon_loss, "forecast": recon_loss.new_zeros(())}
    context = x[..., :-pred_len]
    target = x[..., -pred_len:]
    context_mask = time_mask[..., :-pred_len]
    _, norm_ctx = model.encoder(
        context, context_mask, channel_mask, time_features=None
    )
    normalized = norm_ctx.get("input_normalized", context)
    forecast = model.task_heads.forecast_expert.data_driven(normalized)
    forecast = forecast[..., :channels]
    if forecast.shape[1] != pred_len:
        forecast = forecast[:, :pred_len]
    forecast = forecast * channel_mask[:, None, :].to(forecast.dtype)
    target = target.transpose(1, 2)
    if forecast.shape != target.shape:
        min_pred = min(forecast.shape[1], target.shape[1])
        min_ch = min(forecast.shape[2], target.shape[2])
        forecast = forecast[:, :min_pred, :min_ch]
        target = target[:, :min_pred, :min_ch]
    forecast_loss = F.mse_loss(forecast, target)
    return {"recon": recon_loss, "forecast": forecast_loss}


def build_stage0_loader(config: BasePretrainConfig) -> DataLoader:
    size = 32 if config.smoke else 4000
    dataset: Dataset = RealBackgroundInjectionDataset(
        size=size,
        root=config.data_root,
        seed=config.seed,
        seq_len=config.seq_len,
        max_channels=config.max_channels,
        sources=("ETTm1", "Weather", "TSB-AD", "UCR"),
    )

    def _strip_events(sample: Any) -> Any:
        sample.events_gt = None
        return sample

    class RawOnlyDataset(Dataset):
        def __init__(self, base: Dataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> Any:
            sample = self.base[index]
            sample.events_gt = None
            sample.task_label = {"source": "raw"}
            return sample

    return DataLoader(
        RawOnlyDataset(dataset),
        batch_size=2 if config.smoke else 16,
        shuffle=True,
        collate_fn=padding_collate,
    )


def pretrain_base(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    config: Optional[BasePretrainConfig] = None,
) -> Dict[str, Any]:
    config = config or BasePretrainConfig()
    seed_everything(config.seed)
    device = _device(config.device)
    model.to(device)
    _configure_stage0(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.lr, weight_decay=config.weight_decay
    )
    history = []
    epochs = 1 if config.smoke else config.epochs
    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "recon": 0.0, "forecast": 0.0}
        steps = 0
        for batch in loader:
            batch = _move(batch, device)
            losses = _base_losses(model, batch, config=config)
            total = (
                config.recon_weight * losses["recon"]
                + config.forecast_weight * losses["forecast"]
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            steps += 1
            running["total"] += float(total.detach())
            running["recon"] += float(losses["recon"].detach())
            running["forecast"] += float(losses["forecast"].detach())
            if config.smoke:
                break
        record = {key: value / max(steps, 1) for key, value in running.items()}
        record["epoch"] = epoch + 1
        history.append(record)
        print(json.dumps({"task": "stage0_base", **record}, ensure_ascii=False), flush=True)
    path = Path(config.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "history": history,
            "stage": "0",
        },
        path,
    )
    return {
        "stage": "0",
        "checkpoint": str(path),
        "history": history,
        "loss": history[-1]["total"] if history else None,
    }
