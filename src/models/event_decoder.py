"""Query-based event set decoder."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.types import EventPredictions


class EventDecoder(nn.Module):
    """Decode up to K_max candidate events with existence-aware cross-attention.

    Uses DETR-style object queries with a standard ``TransformerDecoder``. Query
    self-attention is bidirectional (no causal ``tgt_mask``), so slots can
    coordinate over the full candidate set. Existence logits gate active events
    at inference; training may apply sparsity/cardinality regularizers on
    ``existence`` probabilities.
    """

    def __init__(
        self,
        d_model: int,
        max_channels: int,
        num_event_types: int,
        num_attributes: int,
        num_queries: int = 5,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.num_queries = num_queries
        self.max_channels = max_channels
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) / d_model**0.5)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.existence_head = nn.Linear(d_model, 1)
        self.type_head = nn.Linear(d_model, num_event_types)
        self.boundary_head = nn.Linear(d_model, 2)
        self.channel_head = nn.Linear(d_model, max_channels)
        self.attribute_head = nn.Linear(d_model, num_attributes)
        self.attribute_mask_head = nn.Linear(d_model, num_attributes)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(
        self,
        encoded: Tensor,
        time_mask: Tensor,
        channel_mask: Tensor,
        query_override: Optional[Tensor] = None,
    ) -> EventPredictions:
        """Return probabilistic event fields, including ordered boundaries."""

        batch, length, _ = encoded.shape
        channels = channel_mask.shape[1]
        if channels > self.max_channels:
            raise ValueError("channel count exceeds max_channels")
        time_mask = time_mask.to(device=encoded.device, dtype=torch.bool)
        channel_mask = channel_mask.to(device=encoded.device, dtype=torch.bool)
        memory_mask = time_mask.clone()
        empty_rows = ~memory_mask.any(dim=1)
        if length == 0:
            raise ValueError("the temporal dimension must be non-empty")
        memory_mask[empty_rows, 0] = True

        if query_override is None:
            queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        else:
            if query_override.shape[:2] != (batch, self.num_queries):
                raise ValueError("query_override must have shape [B,K,D]")
            queries = query_override
        # No tgt_mask: queries attend bidirectionally (not causal generation).
        decoded = self.decoder(
            tgt=queries,
            memory=encoded,
            memory_key_padding_mask=~memory_mask,
        )
        decoded = self.norm(decoded)
        existence_logits = self.existence_head(decoded).squeeze(-1)
        raw_boundaries = self.boundary_head(decoded)
        start = torch.sigmoid(raw_boundaries[..., 0])
        duration = F.softplus(raw_boundaries[..., 1])
        end = (start + duration).clamp(max=1.0)
        channel_scores = torch.sigmoid(self.channel_head(decoded)[..., :channels])
        channel_scores = channel_scores * channel_mask[:, None, :].to(channel_scores.dtype)
        return {
            "existence_logits": existence_logits,
            "existence": torch.sigmoid(existence_logits),
            "type": torch.softmax(self.type_head(decoded), dim=-1),
            "start": start,
            "end": end,
            "channel_scores": channel_scores,
            "attributes": self.attribute_head(decoded),
            "attribute_mask": torch.sigmoid(self.attribute_mask_head(decoded)),
            "confidence": torch.sigmoid(self.confidence_head(decoded).squeeze(-1)),
            "embeddings": decoded,
        }
