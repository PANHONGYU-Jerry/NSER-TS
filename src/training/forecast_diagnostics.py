"""Forecast protocol audits: naive baselines, scaler checks, single-batch overfit."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.task_metrics import forecast_metrics


def _batch_target(batch: Mapping[str, Any]) -> torch.Tensor:
    for key in ("task_target", "task_label", "target", "y", "label", "labels"):
        value = batch.get(key)
        if value is not None:
            return value
    raise KeyError("Batch needs one of task_target/task_label/target/y/label/labels.")


def _align_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (
        prediction.ndim == 3
        and target.ndim == 3
        and prediction.shape == target.transpose(1, 2).shape
    ):
        return target.transpose(1, 2)
    return target


@torch.no_grad()
def last_value_forecast(history: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Repeat the last observed step for each channel."""

    last = history[..., -1:].expand(-1, -1, pred_len)
    return last.transpose(1, 2)


@torch.no_grad()
def linear_forecast(history: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Extrapolate each channel with a least-squares line over history."""

    batch, channels, length = history.shape
    time = torch.linspace(
        0.0, 1.0, length, device=history.device, dtype=history.dtype
    )
    future = torch.linspace(
        1.0,
        1.0 + pred_len / max(length, 1),
        pred_len,
        device=history.device,
        dtype=history.dtype,
    )
    mean_t = time.mean()
    mean_x = history.mean(dim=-1, keepdim=True)
    slope = (
        ((time - mean_t) * (history - mean_x)).sum(dim=-1, keepdim=True)
        / ((time - mean_t).square().sum().clamp_min(1e-6))
    )
    intercept = mean_x - slope * mean_t
    forecast = intercept + slope * future[None, None, :]
    return forecast.transpose(1, 2).expand(batch, -1, -1)


class _DLinearBaseline(nn.Module):
    """Per-channel trend/seasonal linear map used only for diagnostics."""

    def __init__(self, seq_len: int, pred_len: int, channels: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        kernel = max(3, seq_len // 8)
        if kernel % 2 == 0:
            kernel += 1
        self.register_buffer(
            "avg_kernel",
            torch.ones(1, 1, kernel) / kernel,
            persistent=False,
        )
        self.trend = nn.Linear(seq_len, pred_len)
        self.seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        trend = F.conv1d(
            x,
            self.avg_kernel.expand(x.shape[1], 1, -1),
            padding=self.avg_kernel.shape[-1] // 2,
            groups=x.shape[1],
        )[..., : self.seq_len]
        seasonal = x - trend
        out = self.trend(trend) + self.seasonal(seasonal)
        return out.transpose(1, 2)


def _distribution_stats(
    prediction: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.size == 0:
        return {"pred_mean": float("nan"), "pred_std": float("nan"), "pearson_r": float("nan")}
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denom = np.sqrt((pred_centered**2).sum() * (target_centered**2).sum())
    pearson = float((pred_centered * target_centered).sum() / denom) if denom > 0 else float("nan")
    return {
        "pred_mean": float(pred_flat.mean()),
        "pred_std": float(pred_flat.std()),
        "target_mean": float(target_flat.mean()),
        "target_std": float(target_flat.std()),
        "pearson_r": pearson,
    }


@torch.no_grad()
def run_forecast_baselines(batch: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Evaluate last-value, linear, and untrained DLinear on one batch."""

    history = batch["x"].float()
    target = _batch_target(batch).float()
    target = _align_target(history.transpose(1, 2), target)
    pred_len = target.shape[1]
    results: Dict[str, Dict[str, float]] = {}
    for name, predictor in (
        ("last_value", lambda: last_value_forecast(history, pred_len)),
        ("linear", lambda: linear_forecast(history, pred_len)),
    ):
        prediction = predictor()
        metrics = forecast_metrics(
            target.detach().cpu().numpy(), prediction.detach().cpu().numpy()
        )
        metrics.update(
            _distribution_stats(
                prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
            )
        )
        results[name] = metrics
    dlinear = _DLinearBaseline(history.shape[-1], pred_len, history.shape[1])
    dlinear.eval()
    prediction = dlinear(history)
    metrics = forecast_metrics(
        target.detach().cpu().numpy(), prediction.detach().cpu().numpy()
    )
    metrics.update(
        _distribution_stats(
            prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
        )
    )
    results["dlinear_untrained"] = metrics
    return results


def overfit_single_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    steps: int = 200,
    lr: float = 1e-3,
    device: torch.device,
) -> Dict[str, float]:
    """Train on one batch until MSE approaches zero (implementation sanity check)."""

    model = model.to(device)
    model.train()
    if hasattr(model, "_disable_graph_correction"):
        model._disable_graph_correction = True
    target = _batch_target(batch).float()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    final_mse = float("inf")
    output: Mapping[str, Any] = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["x"],
            batch["time_mask"],
            batch["channel_mask"],
            task="forecast",
            time_features=batch.get("time_features"),
        )
        prediction = output["outputs"]["forecast"]
        aligned = _align_target(prediction, target)
        loss = F.mse_loss(prediction, aligned)
        loss.backward()
        optimizer.step()
        final_mse = float(loss.detach())
    prediction = output["outputs"]["forecast"].detach()
    stats = _distribution_stats(
        prediction.cpu().numpy(), _align_target(prediction, target).cpu().numpy()
    )
    return {"overfit_mse": final_mse, **stats}
