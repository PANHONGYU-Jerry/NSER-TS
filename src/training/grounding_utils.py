"""Helpers for temporal-graph grounding pretraining (Stage 1)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.models.event_graph import LEARNED_RELATIONS, PROGRAM_RELATIONS, RELATION_NAMES
from src.training.losses import program_measurement_consistency_loss


def events_tensor_to_targets(
    events_gt: torch.Tensor,
    *,
    num_types: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """Convert ``[N,5]`` SCM rows to matcher targets."""

    types = events_gt[:, 0].long()
    boundaries = events_gt[:, 1:3]
    channels = torch.zeros(events_gt.shape[0], num_channels, device=events_gt.device)
    channel_ids = events_gt[:, 3].long().clamp(0, num_channels - 1)
    channels.scatter_(1, channel_ids.unsqueeze(1), 1.0)
    attributes = torch.zeros(events_gt.shape[0], 7, device=events_gt.device)
    attributes[:, 0] = events_gt[:, 4]
    attribute_mask = torch.zeros_like(attributes)
    attribute_mask[:, 0] = 1.0
    return {
        "types": types,
        "boundaries": boundaries,
        "channels": channels,
        "attributes": attributes,
        "attribute_mask": attribute_mask,
    }


def batch_events_targets(
    batch_events: Sequence[torch.Tensor | None],
    *,
    num_types: int,
    num_channels: int,
) -> List[Dict[str, torch.Tensor]]:
    targets: List[Dict[str, torch.Tensor]] = []
    for events in batch_events:
        if events is None or events.numel() == 0:
            targets.append(
                {
                    "types": torch.zeros(0, dtype=torch.long),
                    "boundaries": torch.zeros(0, 2),
                    "channels": torch.zeros(0, num_channels),
                }
            )
            continue
        targets.append(
            events_tensor_to_targets(
                events, num_types=num_types, num_channels=num_channels
            )
        )
    return targets


def learned_relation_indices() -> List[int]:
    """Indices of relations that require training (not program-computed)."""

    return [
        index
        for index, name in enumerate(RELATION_NAMES)
        if name in LEARNED_RELATIONS
    ]


def program_relation_mask(num_relations: int, device: torch.device) -> torch.Tensor:
    """Boolean mask over relation channels excluded from supervision."""

    mask = torch.zeros(num_relations, dtype=torch.bool, device=device)
    for name in PROGRAM_RELATIONS:
        if name in RELATION_NAMES:
            mask[RELATION_NAMES.index(name)] = True
    return mask


def relation_supervision_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    existence_mask: torch.Tensor | None = None,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """Mean CE/BCE over valid non-self edges for learnable relations only."""

    learnable = learned_relation_indices()
    if not learnable:
        return predicted_scores.sum() * 0.0

    not_self = ~torch.eye(
        predicted_scores.shape[1], device=predicted_scores.device, dtype=torch.bool
    )[None]
    mask = edge_mask & not_self
    if existence_mask is not None:
        mask = mask & existence_mask[:, :, None] & existence_mask[:, None, :]
    if not mask.any():
        return predicted_scores.sum() * 0.0

    pred = predicted_scores[..., learnable].clamp(1e-6, 1.0 - 1e-6)
    target = target_scores[..., learnable].clamp(0.0, 1.0)
    edge_weights = mask.to(pred.dtype)[..., None]
    pos = target >= 0.5
    neg = ~pos
    weights = torch.where(
        pos,
        edge_weights * pos_weight,
        edge_weights * neg_weight,
    )
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def graph_from_targets(
    model: torch.nn.Module,
    targets: Sequence[Dict[str, torch.Tensor]],
    *,
    num_queries: int,
    num_types: int,
    num_channels: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Build deterministic GT relation scores via the graph builder."""

    batch = len(targets)
    events: Dict[str, torch.Tensor] = {
        "existence": torch.zeros(batch, num_queries, device=device),
        "type": torch.zeros(batch, num_queries, num_types, device=device),
        "start": torch.zeros(batch, num_queries, device=device),
        "end": torch.zeros(batch, num_queries, device=device),
        "channel_scores": torch.zeros(batch, num_queries, num_channels, device=device),
        "attributes": torch.zeros(batch, num_queries, 7, device=device),
        "attribute_mask": torch.zeros(batch, num_queries, 7, device=device),
        "confidence": torch.zeros(batch, num_queries, device=device),
        "embeddings": torch.zeros(batch, num_queries, model.config.d_model, device=device),
    }
    for batch_index, target in enumerate(targets):
        count = min(target["types"].shape[0], num_queries)
        if count == 0:
            continue
        events["existence"][batch_index, :count] = 1.0
        events["type"][batch_index, :count].scatter_(
            1, target["types"][:count, None], 1.0
        )
        events["start"][batch_index, :count] = target["boundaries"][:count, 0]
        events["end"][batch_index, :count] = target["boundaries"][:count, 1]
        events["channel_scores"][batch_index, :count] = target["channels"][:count]
        if "attributes" in target:
            width = min(target["attributes"].shape[-1], 7)
            events["attributes"][batch_index, :count, :width] = target["attributes"][
                :count, :width
            ]
            events["attribute_mask"][batch_index, :count, :width] = 1.0
        events["confidence"][batch_index, :count] = 1.0
    graph = model.graph_builder(events)
    node_mask = events["existence"] >= model.graph_builder.existence_threshold
    not_self = ~torch.eye(num_queries, device=device, dtype=torch.bool)[None]
    edge_mask = node_mask[:, :, None] & node_mask[:, None, :] & not_self
    return graph, edge_mask


CORRUPTION_MODES = (
    "wrong_interval",
    "wrong_type",
    "wrong_channel",
    "wrong_amplitude",
    "drop_event",
    "extra_event",
)


def corrupt_events(
    events_gt: torch.Tensor,
    *,
    rng: random.Random,
    mode: Optional[str] = None,
) -> tuple[torch.Tensor, str]:
    """Return a corrupted copy of ``[N,5]`` event rows and the corruption tag."""

    if events_gt.numel() == 0:
        return events_gt.clone(), "empty"
    corrupted = events_gt.clone()
    chosen = mode or rng.choice(CORRUPTION_MODES)
    if chosen == "wrong_interval" and corrupted.shape[0] > 0:
        index = rng.randrange(corrupted.shape[0])
        shift = rng.uniform(-0.25, 0.25)
        corrupted[index, 1] = (corrupted[index, 1] + shift).clamp(0.0, 0.95)
        corrupted[index, 2] = (corrupted[index, 2] + shift + 0.05).clamp(
            corrupted[index, 1].item() + 0.02, 1.0
        )
    elif chosen == "wrong_type":
        index = rng.randrange(corrupted.shape[0])
        corrupted[index, 0] = (corrupted[index, 0].long() + rng.randint(1, 4)) % 9
    elif chosen == "wrong_channel":
        index = rng.randrange(corrupted.shape[0])
        corrupted[index, 3] = (corrupted[index, 3].long() + 1) % max(
            1, int(corrupted[:, 3].max().item()) + 1
        )
    elif chosen == "wrong_amplitude":
        index = rng.randrange(corrupted.shape[0])
        corrupted[index, 4] *= rng.choice([-1.0, 0.25, 2.5])
    elif chosen == "drop_event" and corrupted.shape[0] > 1:
        index = rng.randrange(corrupted.shape[0])
        corrupted = torch.cat(
            [corrupted[:index], corrupted[index + 1 :]], dim=0
        )
    elif chosen == "extra_event":
        template = corrupted[rng.randrange(corrupted.shape[0])].clone()
        template[1] = rng.uniform(0.0, 0.7)
        template[2] = min(1.0, float(template[1]) + rng.uniform(0.05, 0.2))
        corrupted = torch.cat([corrupted, template.unsqueeze(0)], dim=0)
    return corrupted, chosen


def _score_event_graph(
    model: torch.nn.Module,
    x: torch.Tensor,
    time_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    events_gt: torch.Tensor,
    *,
    num_types: int,
    num_channels: int,
) -> torch.Tensor:
    device = x.device
    h0, _ = model.encoder(x, time_mask, channel_mask)
    target = events_tensor_to_targets(
        events_gt, num_types=num_types, num_channels=num_channels
    )
    slots = model.config.num_queries
    events: Dict[str, torch.Tensor] = {
        "existence": torch.zeros(1, slots, device=device),
        "type": torch.zeros(1, slots, num_types, device=device),
        "start": torch.zeros(1, slots, device=device),
        "end": torch.zeros(1, slots, device=device),
        "channel_scores": torch.zeros(1, slots, num_channels, device=device),
        "attributes": torch.zeros(1, slots, 7, device=device),
        "attribute_mask": torch.zeros(1, slots, 7, device=device),
        "confidence": torch.zeros(1, slots, device=device),
    }
    count = min(target["types"].shape[0], slots)
    if count == 0:
        return x.new_zeros(())
    events["existence"][0, :count] = 1.0
    events["type"][0, :count].scatter_(1, target["types"][:count, None], 1.0)
    events["start"][0, :count] = target["boundaries"][:count, 0]
    events["end"][0, :count] = target["boundaries"][:count, 1]
    events["channel_scores"][0, :count] = target["channels"][:count]
    events["confidence"][0, :count] = 1.0
    field_embeddings = model._event_field_embeddings(events, h0.detach(), time_mask)
    events["embeddings"] = field_embeddings
    graph = model.graph_builder(events)
    reasoned = model.graph_reasoner(field_embeddings, graph)
    return model.compat_scorer.score_from_nodes(reasoned, graph)[0]


def compatibility_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    num_types: int,
    num_channels: int,
    temperature: float = 1.0,
    rng: random.Random,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Softplus ranking loss with L2-normalized graph embeddings."""

    x = batch["x"]
    time_mask = batch["time_mask"]
    channel_mask = batch["channel_mask"]
    positives, negatives = [], []
    events_batches = batch.get("events_list") or batch.get("events_gt") or []
    for index, events in enumerate(events_batches):
        if events is None or events.numel() == 0:
            continue
        events = events.to(x.device)
        positives.append(
            _score_event_graph(
                model,
                x[index : index + 1],
                time_mask[index : index + 1],
                channel_mask[index : index + 1],
                events,
                num_types=num_types,
                num_channels=num_channels,
            )
        )
        corrupted, _ = corrupt_events(events, rng=rng)
        negatives.append(
            _score_event_graph(
                model,
                x[index : index + 1],
                time_mask[index : index + 1],
                channel_mask[index : index + 1],
                corrupted,
                num_types=num_types,
                num_channels=num_channels,
            )
        )
    if not positives:
        return x.sum() * 0.0, {}
    positive = torch.stack(positives)
    negative = torch.stack(negatives)
    loss = F.softplus(-(positive - negative) / max(temperature, 1e-4)).mean()
    diagnostics = {
        "positive_score_mean": float(positive.mean().detach()),
        "positive_score_std": float(positive.std(unbiased=False).detach())
        if positive.numel() > 1
        else 0.0,
        "negative_score_mean": float(negative.mean().detach()),
        "negative_score_std": float(negative.std(unbiased=False).detach())
        if negative.numel() > 1
        else 0.0,
        "positive_negative_margin": float((positive - negative).mean().detach()),
    }
    return loss, diagnostics


@torch.no_grad()
def compatibility_metrics(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    num_types: int,
    num_channels: int,
    rng: random.Random,
) -> Dict[str, float]:
    """AUROC, ranking accuracy, and corruption-type accuracy for Stage 1 gates."""

    x = batch["x"]
    time_mask = batch["time_mask"]
    channel_mask = batch["channel_mask"]
    pos_scores, neg_scores = [], []
    events_batches = batch.get("events_list") or batch.get("events_gt") or []
    for events in events_batches:
        if events is None or events.numel() == 0:
            continue
        events = events.to(x.device)
        pos_scores.append(
            float(
                _score_event_graph(
                    model,
                    x[:1],
                    time_mask[:1],
                    channel_mask[:1],
                    events,
                    num_types=num_types,
                    num_channels=num_channels,
                )
            )
        )
        corrupted, _ = corrupt_events(events, rng=rng)
        neg_scores.append(
            float(
                _score_event_graph(
                    model,
                    x[:1],
                    time_mask[:1],
                    channel_mask[:1],
                    corrupted,
                    num_types=num_types,
                    num_channels=num_channels,
                )
            )
        )
    if not pos_scores:
        return {}
    ranking_acc = sum(
        1 for pos, neg in zip(pos_scores, neg_scores) if pos > neg
    ) / len(pos_scores)
    pos_tensor = torch.tensor(pos_scores)
    neg_tensor = torch.tensor(neg_scores)
    all_scores = torch.tensor(pos_scores + neg_scores)
    all_labels = torch.tensor([1.0] * len(pos_scores) + [0.0] * len(neg_scores))
    sorted_indices = torch.argsort(all_scores)
    sorted_labels = all_labels[sorted_indices]
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1.0 - sorted_labels, dim=0)
    tpr = tps / sorted_labels.sum().clamp_min(1.0)
    fpr = fps / (1.0 - sorted_labels).sum().clamp_min(1.0)
    auroc = float(torch.trapz(tpr, fpr))
    return {
        "compat_auroc": auroc,
        "compat_ranking_acc": float(ranking_acc),
        "corruption_type_acc": float(ranking_acc),
        "positive_score_mean": float(pos_tensor.mean()),
        "negative_score_mean": float(neg_tensor.mean()),
        "positive_negative_margin": float((pos_tensor - neg_tensor).mean()),
    }


def encoder_retain_loss(
    current_encoded: torch.Tensor,
    teacher_encoded: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep encoder outputs close to the frozen Stage-0 teacher."""

    mask = time_mask.unsqueeze(-1).to(current_encoded.dtype)
    diff = (current_encoded - teacher_encoded.detach()).square() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)
