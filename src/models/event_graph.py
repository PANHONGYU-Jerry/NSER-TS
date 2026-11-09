"""Construction of deterministic and learned event relations."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

from src.types import EventGraphData, EventPredictions


def mask_relation_channels(
    graph: EventGraphData,
    indices: Sequence[int],
) -> EventGraphData:
    """Zero selected relation channels (eval-only leave-one-relation-out).

    Used by the Edge contribution heatmap: drop one deterministic relation
    (or the learned-edge block) and measure the F1 drop vs the intact graph.
    """

    if not indices:
        return graph
    scores = graph["relation_scores"].clone()
    num_relations = int(scores.shape[-1])
    for index in indices:
        idx = int(index)
        if idx < 0 or idx >= num_relations:
            raise IndexError(
                f"relation index {idx} out of range [0, {num_relations})"
            )
        scores[..., idx] = 0.0
    return {
        "relation_scores": scores,
        "edge_mask": graph["edge_mask"],
        "edge_weight": graph["edge_weight"],
        "node_mask": graph["node_mask"],
        "existence_probs": graph["existence_probs"],
        "relation_names": list(graph["relation_names"]),
    }


def shuffle_relation_vectors(
    graph: EventGraphData,
    *,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> tuple[EventGraphData, dict]:
    """Permute full relation vectors among valid directed edges.

    Event nodes, ``edge_mask``, ``edge_weight`` and existence stay fixed, so
    the graph topology and existence bias are unchanged. Only the 10-D
    relation payload ``r_uv`` is reassigned. Self-loops are never shuffled.

    The returned stats record how many valid edges actually received a
    different vector. Identical payloads make a permutation a no-op, and that
    rate should be reported with the classification metrics.
    """

    scores = graph["relation_scores"]
    edge_mask = graph["edge_mask"]
    batch, nodes, _, _ = scores.shape
    device = scores.device
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    shuffled_scores = scores.clone()
    n_edges = 0
    n_changed = 0
    for batch_index in range(batch):
        valid = edge_mask[batch_index].bool().clone()
        if valid.shape[0] == nodes:
            valid.fill_diagonal_(False)
        pairs = valid.nonzero(as_tuple=False)
        count = int(pairs.size(0))
        if count < 2:
            n_edges += count
            continue
        perm = torch.randperm(count, generator=generator).to(device)
        src = pairs[:, 0]
        tgt = pairs[:, 1]
        original = scores[batch_index, src, tgt]
        donor = pairs[perm]
        replaced = scores[batch_index, donor[:, 0], donor[:, 1]]
        shuffled_scores[batch_index, src, tgt] = replaced
        n_edges += count
        n_changed += int((~torch.isclose(original, replaced).all(dim=-1)).sum().item())

    change_rate = float(n_changed) / float(n_edges) if n_edges else 0.0
    perturbed = {
        "relation_scores": shuffled_scores,
        "edge_mask": edge_mask,
        "edge_weight": graph["edge_weight"],
        "node_mask": graph["node_mask"],
        "existence_probs": graph["existence_probs"],
        "relation_names": list(graph["relation_names"]),
    }
    stats = {
        "n_edges": int(n_edges),
        "n_changed": int(n_changed),
        "change_rate": change_rate,
    }
    return perturbed, stats


def shuffle_edge_pairs(
    graph: EventGraphData,
    *,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> EventGraphData:
    """Structure control: permute edge payloads among valid directed pairs.

    Keeps ``node_mask`` / existence fixed and preserves the multiset of
    relation vectors (and edge weights) among candidate pairs, but destroys
    which event pair carries which relation. Used to prove that downstream
    gains come from *specific* grounded edges rather than graph capacity alone.
    """

    scores = graph["relation_scores"]
    edge_weight = graph["edge_weight"]
    node_mask = graph["node_mask"]
    batch, nodes, _, _ = scores.shape
    device = scores.device
    # CPU generator keeps the control portable across CUDA/CPU builds.
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    shuffled_scores = scores.clone()
    shuffled_weight = edge_weight.clone()
    eye = torch.eye(nodes, device=device, dtype=torch.bool)
    for batch_index in range(batch):
        candidates = (
            node_mask[batch_index][:, None]
            & node_mask[batch_index][None, :]
            & ~eye
        )
        src, tgt = candidates.nonzero(as_tuple=True)
        count = int(src.numel())
        if count < 2:
            continue
        perm = torch.randperm(count, generator=generator).to(device)
        pair_scores = scores[batch_index, src, tgt]
        pair_weight = edge_weight[batch_index, src, tgt]
        shuffled_scores[batch_index, src, tgt] = pair_scores[perm]
        shuffled_weight[batch_index, src, tgt] = pair_weight[perm]

    return {
        "relation_scores": shuffled_scores,
        "edge_mask": graph["edge_mask"],
        "edge_weight": shuffled_weight,
        "node_mask": node_mask,
        "existence_probs": graph["existence_probs"],
        "relation_names": list(graph["relation_names"]),
    }


DETERMINISTIC_RELATIONS: List[str] = [
    "BEFORE",
    "OVERLAP",
    "CONTAINS",
    "LONGER_THAN",
    "HIGHER_THAN",
    "STEEPER_THAN",
    "SAME_FREQUENCY",
]
LEARNED_RELATIONS: List[str] = ["SUPPORTS", "CONTRADICTS", "TASK_RELEVANT"]
PROGRAM_RELATIONS = DETERMINISTIC_RELATIONS
RELATION_NAMES = DETERMINISTIC_RELATIONS + LEARNED_RELATIONS


class EventGraphBuilder(nn.Module):
    """Build a dense directed graph over a small event set.

    Attribute positions follow V1: amplitude, slope, duration, delta_mean,
    prominence, frequency, phase. Missing attributes disable their relations.
    """

    def __init__(
        self,
        d_model: int,
        existence_threshold: float = 0.5,
        comparison_epsilon: float = 1e-4,
        frequency_tolerance: float = 0.1,
        relation_threshold: float = 0.0,
        disable_attribute_relations: bool = False,
    ) -> None:
        super().__init__()
        self.existence_threshold = existence_threshold
        self.comparison_epsilon = comparison_epsilon
        self.frequency_tolerance = frequency_tolerance
        self.relation_threshold = relation_threshold
        self.disable_attribute_relations = disable_attribute_relations
        pair_dim = 4 * d_model
        self.learned_relations = nn.Sequential(
            nn.Linear(pair_dim, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, len(LEARNED_RELATIONS)),
        )

    @staticmethod
    def _attribute(events: EventPredictions, index: int) -> tuple[Tensor, Tensor]:
        attributes = events["attributes"]
        masks = events["attribute_mask"]
        if attributes.shape[-1] <= index:
            shape = attributes.shape[:-1]
            return attributes.new_zeros(shape), torch.zeros(
                shape, device=attributes.device, dtype=torch.bool
            )
        return attributes[..., index], masks[..., index] >= 0.5

    def forward(self, events: EventPredictions) -> EventGraphData:
        """Derive all relation scores from decoded event fields."""

        embeddings = events["embeddings"]
        batch, nodes, dim = embeddings.shape
        existence_probs = events["existence"].to(embeddings.dtype)
        node_mask = existence_probs >= self.existence_threshold
        not_self = ~torch.eye(nodes, device=embeddings.device, dtype=torch.bool)[None]
        edge_weight = (
            existence_probs[:, :, None] * existence_probs[:, None, :] * not_self.to(existence_probs.dtype)
        )
        edge_mask = node_mask[:, :, None] & node_mask[:, None, :] & not_self

        start_i, start_j = events["start"][:, :, None], events["start"][:, None, :]
        end_i, end_j = events["end"][:, :, None], events["end"][:, None, :]
        duration_i, duration_j = end_i - start_i, end_j - start_j
        eps = self.comparison_epsilon
        before = end_i < start_j
        overlap = (start_i <= end_j) & (start_j <= end_i)
        contains = (start_i <= start_j) & (end_i >= end_j)
        longer = duration_i > duration_j + eps

        height, height_valid = self._attribute(events, 0)
        slope, slope_valid = self._attribute(events, 1)
        frequency, frequency_valid = self._attribute(events, 5)
        higher = (height[:, :, None] > height[:, None, :] + eps) & (
            height_valid[:, :, None] & height_valid[:, None, :]
        )
        steeper = (
            slope.abs()[:, :, None] > slope.abs()[:, None, :] + eps
        ) & (slope_valid[:, :, None] & slope_valid[:, None, :])
        same_frequency = (
            (frequency[:, :, None] - frequency[:, None, :]).abs()
            <= self.frequency_tolerance
        ) & (frequency_valid[:, :, None] & frequency_valid[:, None, :])

        deterministic = torch.stack(
            [before, overlap, contains, longer, higher, steeper, same_frequency],
            dim=-1,
        ).to(embeddings.dtype)
        if self.disable_attribute_relations:
            # HIGHER_THAN / STEEPER_THAN / SAME_FREQUENCY (indices 4..6) are only
            # as reliable as the predicted attributes; drop them so a wrong
            # amplitude/slope/frequency estimate cannot inject false edges.
            deterministic = deterministic.clone()
            deterministic[..., 4:] = 0.0
        source = embeddings[:, :, None, :].expand(batch, nodes, nodes, dim)
        target = embeddings[:, None, :, :].expand(batch, nodes, nodes, dim)
        pair_features = torch.cat(
            [source, target, source - target, source * target], dim=-1
        )
        learned = torch.sigmoid(self.learned_relations(pair_features))
        if self.relation_threshold > 0:
            # Learned relations are near-dense; sparsify so SUPPORTS/CONTRADICTS/
            # TASK_RELEVANT only pass when the pair MLP is confident.
            learned = learned * (learned >= self.relation_threshold).to(learned.dtype)
        relation_scores = torch.cat([deterministic, learned], dim=-1)
        relation_scores = relation_scores * edge_weight[..., None]
        return {
            "relation_scores": relation_scores,
            "edge_mask": edge_mask,
            "edge_weight": edge_weight,
            "node_mask": node_mask,
            "existence_probs": existence_probs,
            "relation_names": list(RELATION_NAMES),
        }
