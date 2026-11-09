"""Project reasoned event-graph nodes back onto the temporal axis."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.types import EventGraphData, EventPredictions


class GraphToTemporal(nn.Module):
    """Map graph node states to a per-timestep latent correction ``Delta H_G``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, d_model)

    def forward(
        self,
        reasoned_events: Tensor,
        events: EventPredictions,
        time_mask: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        batch, length, _ = time_mask.shape[0], time_mask.shape[1], reasoned_events.shape[-1]
        device, dtype = reasoned_events.device, reasoned_events.dtype
        positions = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
        start = events["start"][..., None]
        end = events["end"][..., None]
        membership = torch.sigmoid(40.0 * (positions - start))
        membership = membership * torch.sigmoid(40.0 * (end - positions))
        membership = membership * events["existence"][..., None]
        if not self.training:
            membership = membership * node_mask[..., None].to(membership.dtype)
        membership = membership * time_mask[:, None, :].to(membership.dtype)
        weights = membership / membership.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bkt,bkd->btd", weights, reasoned_events)
        delta = self.project(pooled)
        return delta * time_mask[..., None].to(delta.dtype)
