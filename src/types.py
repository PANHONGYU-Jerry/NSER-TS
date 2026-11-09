"""Shared type declarations for the grounded event graph reasoner."""

from __future__ import annotations

from typing import Dict, List, TypedDict

from torch import Tensor


class NormalizationStats(TypedDict):
    """Statistics required to recover values from instance normalization."""

    mean: Tensor
    scale: Tensor
    time_mask: Tensor
    channel_mask: Tensor


class EventPredictions(TypedDict, total=False):
    """Batched, differentiable event predictions."""

    existence_logits: Tensor
    existence: Tensor
    type: Tensor
    start: Tensor
    end: Tensor
    channel_scores: Tensor
    attributes: Tensor
    attribute_mask: Tensor
    confidence: Tensor
    embeddings: Tensor


class EventGraphData(TypedDict):
    """Dense event graph representation used by the graph transformer."""

    relation_scores: Tensor
    edge_mask: Tensor
    edge_weight: Tensor
    node_mask: Tensor
    existence_probs: Tensor
    relation_names: List[str]


TensorDict = Dict[str, Tensor]


class ReasonerOutput(TypedDict, total=False):
    """Structured result returned by :class:`GroundedEventGraphReasoner`."""

    encoded: Tensor
    normalization: NormalizationStats
    events: EventPredictions
    graph: EventGraphData
    reasoned_events: Tensor
    outputs: TensorDict
    decoder_events: EventPredictions
