"""Pure PyTorch edge-aware graph transformer."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from src.types import EventGraphData

# Eval-time component ablations for event_value diagnostics.
# Set ``GraphReasoner.ablate`` (or per-layer) to one of these strings.
ABLATE_MODES = (
    "",  # full model
    "no_edge_bias",
    "no_edge_message",
    "no_edges",  # node self-attention only (no relation bias/message)
    "attn_only",  # skip FFN
    "ffn_only",  # skip attention; per-node MLP only
    "layer1_only",  # run only the first layer (handled in GraphReasoner)
)


class EdgeAwareGraphTransformerLayer(nn.Module):
    """Multi-head node attention with relation-dependent bias and messages."""

    def __init__(
        self,
        d_model: int,
        num_relations: int,
        nhead: int,
        dropout: float,
        hard_edges: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_bias = nn.Linear(num_relations, nhead, bias=False)
        self.edge_message = nn.Linear(num_relations, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        # When set, training also uses the hard existence-threshold edges, so the
        # train and eval graphs are identical (no soft/hard mask mismatch).
        self.hard_edges = hard_edges
        # Optional per-layer override; normally inherit from GraphReasoner.ablate.
        self.ablate: str = ""

    def forward(
        self,
        nodes: Tensor,
        graph: EventGraphData,
        ablate: Optional[str] = None,
    ) -> Tensor:
        mode = (ablate if ablate is not None else self.ablate) or ""
        batch, count, dim = nodes.shape
        node_mask = graph["node_mask"]
        existence_probs = graph.get("existence_probs")
        if existence_probs is None:
            existence_probs = node_mask.to(nodes.dtype)

        if mode == "ffn_only":
            # Per-node MLP only: no cross-event mixing.
            nodes = self.norm2(nodes + self.dropout(self.feed_forward(nodes)))
            if self.training:
                return nodes * existence_probs[..., None]
            return nodes * node_mask[..., None].to(nodes.dtype)

        q, k, v = self.qkv(nodes).chunk(3, dim=-1)
        q = q.view(batch, count, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch, count, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch, count, self.nhead, self.head_dim).transpose(1, 2)
        relations = graph["relation_scores"]
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        use_edge_bias = mode not in {"no_edge_bias", "no_edges"}
        if use_edge_bias:
            scores = scores + self.edge_bias(relations).permute(0, 3, 1, 2)

        self_edges = torch.eye(count, device=nodes.device, dtype=torch.bool)[None]
        if self.training and not self.hard_edges:
            key_bias = (1.0 - existence_probs) * -1e4
            scores = scores + key_bias[:, None, None, :]
            edge_weight = graph.get("edge_weight")
            if edge_weight is None:
                edge_weight = existence_probs[:, :, None] * existence_probs[:, None, :]
            valid_edges = edge_weight > 1e-6
            valid_edges = valid_edges | (
                self_edges & (existence_probs[:, :, None] > 1e-6)
            )
        else:
            valid_edges = graph["edge_mask"] | (self_edges & node_mask[:, :, None])
        valid = valid_edges[:, None, :, :]
        masked_scores = scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(masked_scores.float(), dim=-1).to(scores.dtype)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        use_edge_message = mode not in {"no_edge_message", "no_edges"}
        if use_edge_message:
            relation_message = self.edge_message(relations)
            relation_message = relation_message.view(
                batch, count, count, self.nhead, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            values = v[:, :, None, :, :] + relation_message
        else:
            values = v[:, :, None, :, :]
        attended = (weights[..., None] * values).sum(dim=3)
        attended = attended.transpose(1, 2).reshape(batch, count, dim)
        nodes = self.norm1(nodes + self.dropout(self.output(attended)))
        if mode != "attn_only":
            nodes = self.norm2(nodes + self.dropout(self.feed_forward(nodes)))
        if self.training:
            return nodes * existence_probs[..., None]
        return nodes * node_mask[..., None].to(nodes.dtype)


class GraphReasoner(nn.Module):
    """Two-layer edge-aware graph transformer."""

    def __init__(
        self,
        d_model: int,
        num_relations: int = 10,
        nhead: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
        hard_edges: bool = False,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("Grounded Event Graph Reasoner V1 uses exactly two layers")
        self.layers = nn.ModuleList(
            [
                EdgeAwareGraphTransformerLayer(
                    d_model=d_model,
                    num_relations=num_relations,
                    nhead=nhead,
                    dropout=dropout,
                    hard_edges=hard_edges,
                )
                for _ in range(num_layers)
            ]
        )
        # Eval-time ablation mode (see ABLATE_MODES). Empty = full model.
        self.ablate: str = ""

    def forward(self, nodes: Tensor, graph: EventGraphData) -> Tensor:
        """Reason over event nodes while respecting graph padding masks."""

        mode = self.ablate or ""
        layers = self.layers
        if mode == "layer1_only":
            layers = self.layers[:1]
        # layer1_only still uses full edge/attn/ffn inside that one layer.
        layer_mode = "" if mode == "layer1_only" else mode
        for layer in layers:
            nodes = layer(nodes, graph, ablate=layer_mode)
        return nodes
