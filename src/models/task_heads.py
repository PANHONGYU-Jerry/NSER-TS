"""V1 task heads: shared decoder D0 on H0 (base) and H' (graph path)."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class TaskHeads(nn.Module):
    """Morphology, event classification, anomaly and forecasting heads."""

    SUPPORTED_TASKS = {
        "morphology",
        "event_classification",
        "anomaly",
        "forecast",
        "all",
    }

    def __init__(
        self,
        d_model: int,
        max_channels: int,
        pred_len: int,
        num_morphology_classes: int,
        num_event_classes: int,
        dropout: float = 0.1,
        separate_graph_head: bool = False,
    ) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.pred_len = pred_len
        combined_dim = 2 * d_model
        self.morphology = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, num_morphology_classes),
        )
        self.event_classification = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, num_event_classes),
        )
        # Optional dedicated graph-path heads. Sharing one Linear across the base
        # input [pool(H0), event_pool] and the graph input
        # [pool(H0+alpha*deltaH), event_pool] forces a single weight matrix to fit
        # two different feature distributions; the base/final double CE then pulls
        # it toward the base path and damps the graph signal. Separate heads keep
        # the two objectives from fighting (analysis of log/20260909_192228).
        self.morphology_graph: Optional[nn.Sequential] = None
        self.event_classification_graph: Optional[nn.Sequential] = None
        if separate_graph_head:
            self.morphology_graph = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(dropout),
                nn.Linear(combined_dim, num_morphology_classes),
            )
            self.event_classification_graph = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(dropout),
                nn.Linear(combined_dim, num_event_classes),
            )
        self.anomaly = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.forecast = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, pred_len * max_channels),
        )

    def _graph_disabled(
        self,
        disable_graph_correction: bool,
        graph_alpha: Optional[Tensor],
    ) -> bool:
        if disable_graph_correction:
            return True
        if graph_alpha is None:
            return False
        return float(graph_alpha.detach().abs()) < 1e-8

    def forward(
        self,
        encoded_base: Tensor,
        encoded_graph: Tensor,
        reasoned_events: Tensor,
        time_mask: Tensor,
        node_mask: Tensor,
        channel_mask: Tensor,
        task: str,
        events: Optional[Mapping[str, Tensor]] = None,
        normalized_input: Optional[Tensor] = None,
        reconstruction_base: Optional[Tensor] = None,
        reconstruction_graph: Optional[Tensor] = None,
        graph_alpha: Optional[Tensor] = None,
        disable_graph_correction: bool = False,
        events_disabled: bool = False,
        graph_residual_enabled: bool = True,
        pool_events: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del normalized_input, reconstruction_base, reconstruction_graph
        if task not in self.SUPPORTED_TASKS:
            choices = ", ".join(sorted(self.SUPPORTED_TASKS))
            raise ValueError(f"unsupported task {task!r}; expected one of {choices}")

        disable = not graph_residual_enabled or self._graph_disabled(
            disable_graph_correction, graph_alpha
        )
        temporal_base = _masked_mean(encoded_base, time_mask, dim=1)
        temporal_graph = _masked_mean(encoded_graph, time_mask, dim=1)
        if events_disabled:
            # True encoder-only ablation (Phase-2 A3 / claim C1): the head sees
            # no event information. The second half of the head input is zeroed
            # rather than time-pooled so encoder_only strictly removes the event
            # pathway; graph correction is also disabled at the source.
            disable = True
            event_pool = torch.zeros_like(temporal_base)
        else:
            # pool_events lets callers feed the head the raw decoder event
            # embeddings instead of the graph-reasoned nodes, so the graph only
            # contributes through the temporal residual instead of being injected
            # twice (pool + deltaH) — the double-channel diagnosis.
            pool_source = reasoned_events if pool_events is None else pool_events
            if events is not None and self.training:
                pool_weight = events["existence"]
            else:
                pool_weight = node_mask.to(pool_source.dtype)
            event_pool = _masked_mean(pool_source, pool_weight, dim=1)
        graph_context = event_pool if not disable else temporal_graph
        combined_base = torch.cat([temporal_base, event_pool], dim=-1)
        combined_graph = torch.cat([temporal_graph, graph_context], dim=-1)

        outputs: Dict[str, Tensor] = {}
        if task in {"morphology", "all"}:
            base = self.morphology(combined_base)
            outputs["base/morphology"] = base
            if disable:
                outputs["morphology"] = base
            else:
                graph_head = self.morphology_graph or self.morphology
                outputs["morphology"] = graph_head(combined_graph)

        if task in {"event_classification", "all"}:
            base = self.event_classification(combined_base)
            outputs["base/event_classification"] = base
            if disable:
                outputs["event_classification"] = base
            else:
                graph_head = self.event_classification_graph or self.event_classification
                outputs["event_classification"] = graph_head(combined_graph)

        if task in {"anomaly", "all"}:
            if events is None:
                context = event_pool[:, None, :].expand(-1, encoded_base.shape[1], -1)
            else:
                positions = torch.linspace(
                    0.0,
                    1.0,
                    encoded_base.shape[1],
                    device=encoded_base.device,
                    dtype=encoded_base.dtype,
                )
                start = events["start"][..., None]
                end = events["end"][..., None]
                membership = torch.sigmoid(40.0 * (positions - start))
                membership = membership * torch.sigmoid(40.0 * (end - positions))
                membership = membership * events["existence"][..., None]
                if not self.training:
                    membership = membership * node_mask[..., None].to(membership.dtype)
                weights = membership.transpose(1, 2)
                context = torch.einsum("btk,bkd->btd", weights, reasoned_events)
                context = context / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            if disable:
                context = temporal_base[:, None, :].expand(-1, encoded_base.shape[1], -1)
            base_features = torch.cat([encoded_base, temporal_base[:, None, :].expand_as(encoded_base)], dim=-1)
            graph_features = torch.cat([encoded_graph, context], dim=-1)
            base_score = torch.sigmoid(self.anomaly(base_features).squeeze(-1))
            graph_score = torch.sigmoid(self.anomaly(graph_features).squeeze(-1))
            mask = time_mask.to(base_score.dtype)
            outputs["base/anomaly"] = base_score * mask
            outputs["anomaly"] = (base_score if disable else graph_score) * mask

        if task in {"forecast", "all"}:
            channels = channel_mask.shape[1]
            base = self.forecast(combined_base).view(
                encoded_base.shape[0], self.pred_len, self.max_channels
            )
            graph = self.forecast(combined_graph).view(
                encoded_base.shape[0], self.pred_len, self.max_channels
            )
            base = base[..., :channels] * channel_mask[:, None, :].to(base.dtype)
            graph = graph[..., :channels] * channel_mask[:, None, :].to(graph.dtype)
            outputs["base/forecast"] = base
            outputs["forecast"] = base if disable else graph

        return outputs
