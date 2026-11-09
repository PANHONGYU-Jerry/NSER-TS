"""Graph compatibility scorer for Stage 1 grounding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.types import EventGraphData, EventPredictions


class CompatibilityScorer(nn.Module):
    """Score how well an event graph is supported by the input window."""

    def __init__(self, d_model: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def pool_graph(reasoned_events: Tensor, node_mask: Tensor) -> Tensor:
        """L2-normalized mean pool over active event nodes."""

        weight = node_mask.to(reasoned_events.dtype)
        pooled = (reasoned_events * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def score_from_nodes(
        self, reasoned_events: Tensor, graph: EventGraphData
    ) -> Tensor:
        embedding = self.pool_graph(reasoned_events, graph["node_mask"])
        return self.scorer(embedding).squeeze(-1)

    def forward(
        self,
        reasoned_events: Tensor,
        graph: EventGraphData,
        *,
        events: EventPredictions | None = None,
    ) -> Tensor:
        del events  # reserved for future task-conditioned compatibility
        return self.score_from_nodes(reasoned_events, graph)
