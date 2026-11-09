"""End-to-end Grounded Event Graph Reasoner V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.compatibility_scorer import CompatibilityScorer
from src.models.event_decoder import EventDecoder
from src.models.event_graph import (
    EventGraphBuilder,
    RELATION_NAMES,
    mask_relation_channels,
    shuffle_edge_pairs,
    shuffle_relation_vectors,
)
from src.models.graph_reasoner import GraphReasoner
from src.models.graph_to_temporal import GraphToTemporal
from src.models.task_heads import TaskHeads
from src.models.temporal_encoder import TemporalEncoder
from src.types import EventPredictions, ReasonerOutput


@dataclass(frozen=True)
class GroundedReasonerConfig:
    """Configuration for the V1 model."""

    max_channels: int = 64
    d_model: int = 128
    nhead: int = 4
    dropout: float = 0.1
    num_queries: int = 4
    num_event_types: int = 9
    num_attributes: int = 7
    num_morphology_classes: int = 6
    num_event_classes: int = 6
    pred_len: int = 96
    alpha_max: float = 1.0
    graph_residual_cap: float = 0.1
    patch_len: int = 16
    patch_stride: int = 8
    encoder_layers: int = 3
    d_ff: int = 256
    oracle_events: bool = False
    postprocess_events: bool = False
    existence_threshold: float = 0.5
    # True encoder-only architecture switch (Phase-2 A3 / claim C1): the event
    # decoder still runs so checkpoints/aux machinery stay intact, but the task
    # heads receive a zeroed event pool and the graph residual is disabled.
    events_disabled: bool = False
    # --- Graph-path fixes (root-caused in log/20260909_192228) ----------------
    # graph_residual_enabled=False keeps the graph reasoning in the event pool
    # but removes the temporal correction H0 + alpha*deltaH, isolating the
    # residual from the node-message path.
    graph_residual_enabled: bool = True
    # Where the task head's event pool comes from:
    #   "graph"   graph-reasoned nodes (legacy; graph info enters twice: pool+residual)
    #   "decoder" raw decoder event embeddings, so the graph only acts via deltaH
    graph_event_pool_source: str = "graph"
    # Dedicated head weights for the graph path so the base and graph inputs do
    # not share one weight matrix (avoids the base/final double-CE gradient fight).
    separate_graph_head: bool = False
    # Build graph edges from the hard existence threshold in training too, instead
    # of soft existence-weighted edges (removes the train/eval mask mismatch).
    graph_hard_edges: bool = False
    # Drop learned relation scores below this value (0 disables). Sparsifies the
    # otherwise near-dense SUPPORTS/CONTRADICTS/TASK_RELEVANT edges.
    graph_relation_threshold: float = 0.0
    # Zero out attribute-based deterministic relations (HIGHER_THAN/STEEPER_THAN/
    # SAME_FREQUENCY); they are only as reliable as the predicted attributes.
    graph_disable_attribute_relations: bool = False
    # Initial value of the graph residual gate (alpha = alpha_max * tanh(gate)).
    graph_gate_init: float = 0.1
    # Structure-control negative: after building the grounded event graph, permute
    # relation payloads among valid directed pairs (see shuffle_edge_pairs).
    # Prefer the runtime flag ``model._shuffle_graph_edges`` for eval sweeps.
    shuffle_graph_edges: bool = False
    shuffle_graph_seed: int = 0
    # Relation Perturbation Ablation. Applied after the event graph is built,
    # so connectivity and existence bias stay intact.
    #   original              full r_uv (bias + message)
    #   shuffled              permute r_uv on valid edges
    #   no_relation_bias      drop (w_R)^T r_uv, keep W_RV r_uv
    #   no_relation_features  drop both relation bias and relation message
    # Empty / "original" leaves GraphReasoner.ablate untouched so other
    # eval-time component probes can still set it directly.
    relation_mode: str = "original"
    # Seed for shuffled relations. Negative => a fresh permutation each
    # training step, and seed 0 at eval unless the caller overrides it.
    shuffle_relation_seed: int = -1
    # Skip GraphReasoner entirely: reasoned nodes = field embeddings (event-set
    # identity). Isolates "reasoning over event nodes" from edge topology.
    identity_graph: bool = False
    # When False, skip EdgeAwareGraphTransformer (GraphReasoner) entirely —
    # same forward effect as identity_graph. Use for runs that must not use
    # the edge-aware graph transformer.
    edge_transformer_enabled: bool = True


class GroundedEventGraphReasoner(nn.Module):
    """Ground time-series events, reason over them, and solve four tasks."""

    TASK_ALIASES = {
        "event": "event_classification",
        "classification": "event_classification",
        "forecasting": "forecast",
    }

    def __init__(self, config: GroundedReasonerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TemporalEncoder(
            max_channels=config.max_channels,
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            transformer_layers=config.encoder_layers,
            patch_len=config.patch_len,
            patch_stride=config.patch_stride,
            d_ff=config.d_ff,
        )
        self.event_decoder = EventDecoder(
            d_model=config.d_model,
            max_channels=config.max_channels,
            num_event_types=config.num_event_types,
            num_attributes=config.num_attributes,
            num_queries=config.num_queries,
            nhead=config.nhead,
            dropout=config.dropout,
        )
        event_field_dim = (
            1
            + config.num_event_types
            + 2
            + config.max_channels
            + 2 * config.num_attributes
            + 1
        )
        self.event_field_projection = nn.Sequential(
            nn.Linear(event_field_dim, 2 * config.d_model),
            nn.GELU(),
            nn.Linear(2 * config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.global_event_position = nn.Sequential(
            nn.Linear(4, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.relative_point_position = nn.Sequential(
            nn.Linear(5, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.graph_builder = EventGraphBuilder(
            config.d_model,
            existence_threshold=config.existence_threshold,
            relation_threshold=config.graph_relation_threshold,
            disable_attribute_relations=config.graph_disable_attribute_relations,
        )
        self.graph_reasoner = GraphReasoner(
            d_model=config.d_model,
            num_relations=len(RELATION_NAMES),
            nhead=config.nhead,
            dropout=config.dropout,
            hard_edges=config.graph_hard_edges,
        )
        self.graph_to_temporal = GraphToTemporal(config.d_model)
        self.compat_scorer = CompatibilityScorer(config.d_model)
        self.graph_gate = nn.Parameter(torch.tensor([config.graph_gate_init]))
        self.task_heads = TaskHeads(
            d_model=config.d_model,
            max_channels=config.max_channels,
            pred_len=config.pred_len,
            num_morphology_classes=config.num_morphology_classes,
            num_event_classes=config.num_event_classes,
            dropout=config.dropout,
            separate_graph_head=config.separate_graph_head,
        )
        self.reconstruction_head = nn.Linear(config.d_model, config.max_channels)
        self._disable_graph_correction = False
        # Eval-time structure control (event_value analysis). Mirrors config but
        # stays mutable so a loaded checkpoint can be swept without rebuild.
        self._shuffle_graph_edges = bool(config.shuffle_graph_edges)
        self._shuffle_graph_seed = int(config.shuffle_graph_seed)
        self._identity_graph = bool(config.identity_graph)
        configured_mode = str(config.relation_mode or "original").strip().lower()
        # Leave unset for the default so external GraphReasoner.ablate probes
        # (event_value component diagnosis) are not cleared on every forward.
        self._relation_mode = "" if configured_mode == "original" else configured_mode
        self._shuffle_relation_seed = int(config.shuffle_relation_seed)
        self._last_relation_diag: Dict[str, Any] = {}
        # Edge contribution heatmap: zero selected relation channels before
        # GraphReasoner (leave-one-relation-out). None / empty = intact.
        self._mask_relation_indices: Optional[List[int]] = None

    def graph_alpha(self) -> Tensor:
        """``alpha = alpha_max * tanh(g)``; ``g`` starts small so graph path is trainable."""

        alpha = self.config.alpha_max * torch.tanh(self.graph_gate)
        if (
            getattr(self, "_disable_graph_correction", False)
            or getattr(self.config, "events_disabled", False)
            or not getattr(self.config, "graph_residual_enabled", True)
        ):
            return alpha.new_zeros(())
        return alpha

    def _event_field_embeddings(
        self, events: EventPredictions, encoded: Tensor, time_mask: Tensor
    ) -> Tensor:
        channels = events["channel_scores"]
        if channels.shape[-1] > self.config.max_channels:
            raise ValueError("event channel_scores exceeds max_channels")
        channels = F.pad(
            channels, (0, self.config.max_channels - channels.shape[-1]), value=0.0
        )
        fields = torch.cat(
            [
                events["existence"].unsqueeze(-1),
                events["type"],
                events["start"].unsqueeze(-1),
                events["end"].unsqueeze(-1),
                channels,
                events["attributes"],
                events["attribute_mask"],
                events["confidence"].unsqueeze(-1),
            ],
            dim=-1,
        )
        field_embedding = self.event_field_projection(torch.nan_to_num(fields))

        start, end = events["start"], events["end"]
        center = 0.5 * (start + end)
        duration = (end - start).clamp_min(1e-4)
        global_position = self.global_event_position(
            torch.stack([start, end, center, duration], dim=-1)
        )

        length = encoded.shape[1]
        position = torch.linspace(
            0.0, 1.0, length, device=encoded.device, dtype=encoded.dtype
        )[None, None, :]
        relative = ((position - start[..., None]) / duration[..., None]).clamp(0.0, 1.0)
        relative_features = torch.stack(
            [
                relative,
                torch.sin(torch.pi * relative),
                torch.cos(torch.pi * relative),
                torch.sin(2.0 * torch.pi * relative),
                torch.cos(2.0 * torch.pi * relative),
            ],
            dim=-1,
        )
        local_position = self.relative_point_position(relative_features)
        membership = torch.sigmoid(40.0 * (position - start[..., None]))
        membership = membership * torch.sigmoid(40.0 * (end[..., None] - position))
        membership = membership * time_mask[:, None, :].to(membership.dtype)
        local_values = encoded[:, None, :, :] + local_position
        local_shape = (local_values * membership[..., None]).sum(dim=2)
        local_shape = local_shape / membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return field_embedding + global_position + local_shape

    def _reconstruction_from_encoded(
        self,
        encoded: Tensor,
        normalization: dict,
        channel_mask: Tensor,
        time_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        channels = channel_mask.shape[1]
        reconstruction_normalized = self.reconstruction_head(encoded)[
            ..., :channels
        ].transpose(1, 2)
        reconstruction = (
            reconstruction_normalized * normalization["scale"]
            + normalization["mean"]
        )
        valid = time_mask[:, None, :].to(reconstruction.dtype) * channel_mask[
            :, :, None
        ].to(reconstruction.dtype)
        return reconstruction_normalized, reconstruction * valid

    def _apply_relation_mode(
        self,
        graph: Dict[str, Any],
        relation_mode: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Edit relation inputs after topology is fixed.

        ``no_relation_bias`` removes ``(w_R)^T r_uv`` from the attention logits.
        ``no_relation_features`` also removes ``W_RV r_uv`` from the value
        message. Neighborhood aggregation and existence bias stay.
        ``shuffled`` reassigns the full relation vector across valid edges.
        """

        mode = relation_mode.strip().lower()
        ablate = {
            "original": "",
            "shuffled": "",
            "no_relation_bias": "no_edge_bias",
            "no_relation_features": "no_edges",
        }
        if mode not in ablate:
            raise ValueError(
                "relation_mode must be one of "
                "original, shuffled, no_relation_bias, no_relation_features; "
                f"got {relation_mode!r}"
            )
        self.graph_reasoner.ablate = ablate[mode]
        stats = {"n_edges": 0, "n_changed": 0, "change_rate": 0.0}
        if mode != "shuffled":
            return graph, stats
        seed = int(getattr(self, "_shuffle_relation_seed", -1))
        if seed < 0:
            if self.training:
                seed = int(torch.randint(1, 2**31 - 1, ()).item())
            else:
                seed = 0
        graph, stats = shuffle_relation_vectors(graph, seed=seed)
        return graph, stats

    def _apply_graph_residual(self, encoded: Tensor, delta_h: Tensor) -> Tensor:
        alpha = self.graph_alpha()
        delta_norm = delta_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        h_norm = encoded.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cap = self.config.graph_residual_cap
        scale = torch.minimum(
            torch.ones_like(h_norm),
            cap * h_norm / delta_norm,
        )
        delta_bounded = delta_h * scale
        return encoded + alpha * delta_bounded, delta_bounded, alpha

    def _reason_and_predict(
        self,
        encoded: Tensor,
        normalization: dict,
        events: EventPredictions,
        time_mask: Tensor,
        channel_mask: Tensor,
        task: str,
    ) -> ReasonerOutput:
        canonical_task = self.TASK_ALIASES.get(task, task)
        h0 = encoded
        field_embeddings = self._event_field_embeddings(events, h0, time_mask)
        decoder_embeddings = events.get("embeddings")
        if decoder_embeddings is not None:
            field_embeddings = field_embeddings + decoder_embeddings
        events = dict(events)
        events["embeddings"] = field_embeddings
        graph = self.graph_builder(events)
        mask_indices = getattr(self, "_mask_relation_indices", None)
        if mask_indices:
            graph = mask_relation_channels(graph, list(mask_indices))
        shuffle_edges = bool(
            getattr(self, "_shuffle_graph_edges", False)
            or getattr(self.config, "shuffle_graph_edges", False)
        )
        if shuffle_edges:
            seed = int(
                getattr(self, "_shuffle_graph_seed", getattr(self.config, "shuffle_graph_seed", 0))
            )
            graph = shuffle_edge_pairs(graph, seed=seed)
        relation_stats: Dict[str, Any] = {
            "n_edges": 0,
            "n_changed": 0,
            "change_rate": 0.0,
        }
        relation_mode = str(getattr(self, "_relation_mode", "") or "").strip().lower()
        if relation_mode:
            graph, relation_stats = self._apply_relation_mode(graph, relation_mode)
        # identity_graph / edge_transformer_enabled=False: skip GraphReasoner
        # (EdgeAwareGraphTransformer) so the head / residual see raw field
        # embeddings. Isolates event-set features from edge-aware reasoning.
        skip_edge_transformer = (
            getattr(self, "_identity_graph", False)
            or getattr(self.config, "identity_graph", False)
            or not bool(getattr(self.config, "edge_transformer_enabled", True))
        )
        if skip_edge_transformer:
            reasoned = field_embeddings
        else:
            reasoned = self.graph_reasoner(field_embeddings, graph)
        delta_h = self.graph_to_temporal(
            reasoned, events, time_mask, graph["node_mask"]
        )
        encoded_graph, delta_bounded, alpha = self._apply_graph_residual(h0, delta_h)
        if not self.training:
            ratio = (
                delta_bounded.detach().float().norm(dim=-1)
                / h0.detach().float().norm(dim=-1).clamp_min(1e-6)
            )
            alpha_value = float(alpha.detach().float().reshape(-1)[0].item())
            self._last_relation_diag = {
                "relation_mode": relation_mode or "original",
                "alpha": alpha_value,
                "residual_ratio": float(ratio.mean().item()),
                "relation_change_rate": float(relation_stats.get("change_rate", 0.0)),
                "n_edges": int(relation_stats.get("n_edges", 0)),
                "n_changed": int(relation_stats.get("n_changed", 0)),
            }

        # "decoder" pool source keeps the graph from entering the head twice
        # (event pool + temporal residual): the pool then carries the raw decoder
        # event embeddings and only deltaH carries graph information.
        pool_events = None
        if getattr(self.config, "graph_event_pool_source", "graph") == "decoder":
            pool_events = decoder_embeddings if decoder_embeddings is not None else reasoned

        recon_norm_base, reconstruction_base = self._reconstruction_from_encoded(
            h0, normalization, channel_mask, time_mask
        )
        recon_norm_graph, reconstruction_graph = self._reconstruction_from_encoded(
            encoded_graph, normalization, channel_mask, time_mask
        )
        outputs = self.task_heads(
            encoded_base=h0,
            encoded_graph=encoded_graph,
            reasoned_events=reasoned,
            time_mask=time_mask,
            node_mask=graph["node_mask"],
            channel_mask=channel_mask,
            task=canonical_task,
            events=events,
            reconstruction_base=recon_norm_base,
            reconstruction_graph=recon_norm_graph,
            graph_alpha=alpha,
            disable_graph_correction=getattr(self, "_disable_graph_correction", False),
            events_disabled=bool(getattr(self.config, "events_disabled", False)),
            graph_residual_enabled=bool(
                getattr(self.config, "graph_residual_enabled", True)
            ),
            pool_events=pool_events,
        )
        return {
            "encoded": h0,
            "encoded_graph": encoded_graph,
            "normalization": normalization,
            "events": events,
            "graph": graph,
            "reasoned_events": reasoned,
            "graph_delta": delta_bounded,
            "graph_alpha": alpha,
            "outputs": outputs,
            "reconstruction": reconstruction_graph,
            "reconstruction_base": reconstruction_base,
        }

    def _oracle_events_from_gt(
        self,
        events_gt: Mapping[str, Any],
        batch_size: int,
        device: torch.device,
        decoder_events: Optional[Mapping[str, Any]] = None,
        channel_mask: Optional[Tensor] = None,
    ) -> EventPredictions:
        """Build batched EventPredictions tensors from per-sample GT events.

        When ``decoder_events`` is available, each GT event is aligned to the
        decoder slot with the closest boundary center so the task head keeps
        the decoder's contextual embeddings; only the semantic fields
        (existence/type/boundaries/attributes/confidence) are overridden.
        """
        required = {
            "types",
            "boundaries",
            "channels",
            "event_channels",
            "attributes",
            "attribute_mask",
            "confidence",
        }
        missing = required.difference(events_gt)
        if missing:
            raise ValueError(f"events_gt missing fields: {sorted(missing)}")
        num_queries = int(self.config.num_queries)
        num_types = int(self.config.num_event_types)
        # Channel width must match the event decoder's channel_scores (real
        # channels in this batch), otherwise the EventSetCriterion channel BCE
        # sees target (max_channels) vs input (n_channels) size mismatches.
        num_channels = (
            int(channel_mask.shape[1])
            if channel_mask is not None
            else int(self.config.max_channels)
        )
        num_attributes = int(self.config.num_attributes)
        events: EventPredictions = {
            "existence_logits": torch.full(
                (batch_size, num_queries), -6.0, device=device, dtype=torch.float32
            ),
            "existence": torch.zeros(
                batch_size, num_queries, device=device, dtype=torch.float32
            ),
            "type": torch.zeros(
                batch_size, num_queries, num_types, device=device, dtype=torch.float32
            ),
            "start": torch.zeros(
                batch_size, num_queries, device=device, dtype=torch.float32
            ),
            "end": torch.zeros(
                batch_size, num_queries, device=device, dtype=torch.float32
            ),
            "channel_scores": torch.zeros(
                batch_size, num_queries, num_channels, device=device, dtype=torch.float32
            ),
            "attributes": torch.zeros(
                batch_size, num_queries, num_attributes, device=device, dtype=torch.float32
            ),
            "attribute_mask": torch.zeros(
                batch_size, num_queries, num_attributes, device=device, dtype=torch.float32
            ),
            "confidence": torch.zeros(
                batch_size, num_queries, device=device, dtype=torch.float32
            ),
            "embeddings": (
                decoder_events.get("embeddings") if decoder_events is not None else None
            ),
        }
        decoder_start = None
        decoder_end = None
        if decoder_events is not None:
            decoder_start = decoder_events.get("start")
            decoder_end = decoder_events.get("end")
        for batch_index in range(batch_size):
            count = min(len(events_gt["types"][batch_index]), num_queries)
            if count <= 0:
                continue
            if decoder_start is not None and count <= num_queries:
                gt_centers = (
                    events_gt["boundaries"][batch_index][:count].mean(dim=-1).to(device)
                )
                slot_centers = 0.5 * (
                    decoder_start[batch_index] + decoder_end[batch_index]
                )
                slot_centers = slot_centers[None, :].expand(count, num_queries)
                cost = (gt_centers[:, None] - slot_centers).abs()
                used: set[int] = set()
                slots: List[int] = []
                for _ in range(count):
                    row, column = torch.argmin(cost).item() // num_queries, torch.argmin(
                        cost
                    ).item() % num_queries
                    while column in used:
                        cost[:, column] = float("inf")
                        row, column = torch.argmin(cost).item() // num_queries, torch.argmin(
                            cost
                        ).item() % num_queries
                    slots.append(column)
                    used.add(column)
                    cost[row, :] = float("inf")
                    cost[:, column] = float("inf")
            else:
                slots = list(range(count))
            slot_tensor = torch.tensor(slots, device=device)
            type_indices = events_gt["types"][batch_index][:count].to(device)
            events["existence_logits"][batch_index, slot_tensor] = 6.0
            events["existence"][batch_index, slot_tensor] = 1.0
            events["type"][batch_index, slot_tensor, type_indices] = 1.0
            boundaries = events_gt["boundaries"][batch_index][:count].to(device)
            events["start"][batch_index, slot_tensor] = boundaries[:, 0]
            events["end"][batch_index, slot_tensor] = boundaries[:, 1]
            channel_indices = events_gt["event_channels"][batch_index][:count].to(device)
            channel_indices = channel_indices.clamp(0, num_channels - 1)
            events["channel_scores"][batch_index, slot_tensor, channel_indices] = 1.0
            events["attributes"][batch_index, slot_tensor] = events_gt["attributes"][
                batch_index
            ][:count].to(device)
            events["attribute_mask"][batch_index, slot_tensor] = events_gt["attribute_mask"][
                batch_index
            ][:count].to(device)
            events["confidence"][batch_index, slot_tensor] = events_gt["confidence"][
                batch_index
            ][:count].to(device)
        return events

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor,
        channel_mask: Tensor,
        task: str = "all",
        events_gt: Optional[Mapping[str, Any]] = None,
    ) -> ReasonerOutput:
        encoded, normalization = self.encoder(x, time_mask, channel_mask)
        oracle_mode = bool(getattr(self.config, "oracle_events", False)) and events_gt is not None
        if oracle_mode:
            decoder_events = self.event_decoder(encoded, time_mask, channel_mask)
            events = self._oracle_events_from_gt(
                events_gt,
                batch_size=x.shape[0],
                device=x.device,
                decoder_events=decoder_events,
                channel_mask=channel_mask,
            )
        else:
            events = self.event_decoder(encoded, time_mask, channel_mask)
        if (
            not oracle_mode
            and self.config.postprocess_events
            and not self.training
        ):
            from src.evaluation.event_postprocess import (
                EventPostprocessConfig,
                postprocess_events_tensor,
            )

            events = postprocess_events_tensor(
                events,
                time_mask=time_mask,
                config=EventPostprocessConfig(
                    existence_threshold=self.config.existence_threshold
                ),
            )
        result = self._reason_and_predict(
            encoded,
            normalization,
            events,
            time_mask.to(device=x.device, dtype=torch.bool),
            channel_mask.to(device=x.device, dtype=torch.bool),
            task,
        )
        if oracle_mode:
            result["decoder_events"] = decoder_events
        return result

    def rerun_with_events(
        self,
        x: Tensor,
        time_mask: Tensor,
        channel_mask: Tensor,
        events: Mapping[str, Tensor],
        task: str = "all",
    ) -> ReasonerOutput:
        required = {
            "existence",
            "type",
            "start",
            "end",
            "channel_scores",
            "attributes",
            "attribute_mask",
            "confidence",
        }
        missing = required.difference(events)
        if missing:
            raise ValueError(f"repaired events missing fields: {sorted(missing)}")
        repaired: EventPredictions = {
            key: value.to(device=x.device) for key, value in events.items() if key in required
        }
        repaired["start"], repaired["end"] = (
            torch.minimum(repaired["start"], repaired["end"]).clamp(0.0, 1.0),
            torch.maximum(repaired["start"], repaired["end"]).clamp(0.0, 1.0),
        )
        encoded, normalization = self.encoder(x, time_mask, channel_mask)
        return self._reason_and_predict(
            encoded,
            normalization,
            repaired,
            time_mask.to(device=x.device, dtype=torch.bool),
            channel_mask.to(device=x.device, dtype=torch.bool),
            task,
        )
