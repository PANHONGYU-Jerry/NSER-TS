"""Losses shared by event pretraining and downstream tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher, interval_iou_1d


def existence_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(prediction).float()
    if torch.all((prediction >= 0) & (prediction <= 1)):
        return F.binary_cross_entropy(prediction, target)
    return F.binary_cross_entropy_with_logits(prediction, target)


def type_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if torch.all((prediction >= 0) & (prediction <= 1)):
        return F.nll_loss(prediction.clamp_min(1e-8).log(), target.long())
    return F.cross_entropy(prediction, target.long())


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target.to(prediction))


def iou_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pairwise = interval_iou_1d(prediction, target.to(prediction))
    return 1.0 - pairwise.diagonal(dim1=-2, dim2=-1).mean()


def attribute_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    error = F.smooth_l1_loss(prediction, target.to(prediction), reduction="none")
    if mask is not None:
        valid = mask.to(error).bool()
        return error[valid].mean() if valid.any() else error.sum() * 0.0
    return error.mean()


def channel_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(prediction).float()
    if torch.all((prediction >= 0) & (prediction <= 1)):
        return F.binary_cross_entropy(prediction, target)
    return F.binary_cross_entropy_with_logits(prediction, target)


def task_loss(prediction: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    if task in {"morphology", "event_classification"}:
        return F.cross_entropy(prediction, target.long())
    if task == "anomaly":
        prediction = prediction.squeeze(-1)
        target = target.float()
        if target.shape != prediction.shape and target.ndim == 3:
            target = target.squeeze(1)
        if torch.all((prediction >= 0) & (prediction <= 1)):
            return F.binary_cross_entropy(prediction, target.to(prediction))
        return F.binary_cross_entropy_with_logits(prediction, target.to(prediction))
    if task == "forecast":
        if (
            prediction.ndim == 3
            and target.ndim == 3
            and prediction.shape == target.transpose(1, 2).shape
        ):
            target = target.transpose(1, 2)
        return F.mse_loss(prediction, target.to(prediction))
    raise ValueError(f"Unsupported task loss: {task}")


def measurement_consistency_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    measurement_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep predicted/reconstructed measurements consistent with observed values."""
    lhs = measurement_fn(prediction) if measurement_fn else prediction
    rhs = measurement_fn(reference) if measurement_fn else reference
    error = (lhs - rhs.to(lhs)).abs()
    if mask is not None:
        valid = mask.to(error).bool()
        return error[valid].mean() if valid.any() else error.sum() * 0.0
    return error.mean()


def program_measurement_consistency_loss(
    raw_x: torch.Tensor,
    events: Mapping[str, torch.Tensor],
    time_mask: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compare predicted attributes with independent raw-signal measurements.

    Boundaries/channels are treated as discrete proposals.  Gradients flow only
    into the attribute predictions.
    """

    from src.event_schema import measure_attributes, robust_scale

    attributes = events["attributes"]
    existence = events["existence"].detach()
    starts, ends = events["start"].detach(), events["end"].detach()
    channels = events["channel_scores"].detach().argmax(dim=-1)
    terms = []
    for batch_index in range(raw_x.shape[0]):
        length = int(time_mask[batch_index].sum().item())
        n_channels = int(channel_mask[batch_index].sum().item())
        raw = raw_x[batch_index, :n_channels, :length].detach().cpu().numpy()
        scale = robust_scale(raw)
        for slot in range(attributes.shape[1]):
            event = {
                "start": float(starts[batch_index, slot]),
                "end": float(ends[batch_index, slot]),
                "channel": int(channels[batch_index, slot]),
            }
            measured = measure_attributes(raw, event)
            target = attributes.new_tensor(
                [
                    measured["amplitude"] / scale,
                    measured["slope"] * length / scale,
                    measured["duration"] / max(length, 1),
                    measured["delta_mean"] / scale,
                    measured["prominence"] / scale,
                    measured["frequency"] * length,
                    measured["phase"] / torch.pi,
                ]
            )
            # Windows containing strong anomalies (e.g. Exathlon facility data)
            # can have raw amplitudes orders of magnitude above the robust-scale
            # floor, making the normalized targets explode; clamp so the
            # program-consistency term cannot blow up the total loss.
            target = target.clamp(-5.0, 5.0)
            width = min(attributes.shape[-1], target.numel())
            error = F.smooth_l1_loss(
                attributes[batch_index, slot, :width],
                target[:width],
                reduction="mean",
            )
            terms.append(existence[batch_index, slot] * error)
    return torch.stack(terms).mean() if terms else attributes.sum() * 0.0


def margin_ranking_loss(
    positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Require positive explanations/events to score above negative ones."""
    return F.relu(margin - positive + negative).mean()


def existence_sparsity_loss(existence_probs: torch.Tensor) -> torch.Tensor:
    """L1 penalty encouraging fewer active event candidates."""
    return existence_probs.mean()


def existence_entropy_loss(existence_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy regularizer pushing existence toward 0 or 1."""
    probability = existence_probs.clamp(eps, 1.0 - eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    return entropy.mean()


def event_width_prior_loss(events: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Soft penalty when predicted intervals violate type-specific width priors."""
    from src.evaluation.event_postprocess import event_width_prior_loss as _width_prior

    return _width_prior(events)


def adaptive_cardinality_loss(
    events: Mapping[str, torch.Tensor],
    raw_x: torch.Tensor,
    task: str,
) -> torch.Tensor:
    """Soft target on the number of active event slots."""

    existence = events["existence"]
    max_events = existence.shape[1]
    base = {
        "morphology": 2.0,
        "event_classification": 3.0,
        "anomaly": 3.0,
        "forecast": 3.0,
    }[task]
    with torch.no_grad():
        if raw_x.shape[-1] > 1:
            variation = (raw_x[..., 1:] - raw_x[..., :-1]).abs().mean(dim=(-1, -2))
            scale = raw_x.detach().std(dim=(-1, -2)).clamp_min(1e-6)
            complexity = (variation / scale).clamp(0.0, 2.0)
        else:
            complexity = raw_x.new_zeros(raw_x.shape[0])
        channel_bonus = min(raw_x.shape[1] / 9.0, 1.0)
        target = (
            raw_x.new_full((raw_x.shape[0],), base)
            + 0.5 * complexity
            + 0.5 * channel_bonus
        ).clamp(1.0, float(max_events))
    return (existence.sum(dim=-1) - target).abs().mean()


def anomaly_regularization(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Discourage high scores on normal points and fragmented alarms."""

    prediction = prediction.squeeze(-1)
    target = target.to(prediction).float()
    if target.shape != prediction.shape and target.ndim == 3:
        target = target.squeeze(1)
    normal = target < 0.5
    sparsity = prediction[normal].mean() if normal.any() else prediction.sum() * 0.0
    compactness = (
        (prediction[:, 1:] - prediction[:, :-1]).abs().mean()
        if prediction.ndim == 2 and prediction.shape[1] > 1
        else prediction.sum() * 0.0
    )
    return sparsity + 0.1 * compactness


class EventSetCriterion(nn.Module):
    """DETR-style event loss after Hungarian matching."""

    def __init__(
        self,
        matcher: Optional[HungarianMatcher] = None,
        weights: Optional[Mapping[str, float]] = None,
        no_event_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.matcher = matcher or HungarianMatcher()
        self.weights = {
            "exist": 1.0,
            "type": 1.0,
            "boundary": 2.0,
            "iou": 2.0,
            "attribute": 0.5,
            "channel": 0.5,
            "confidence": 0.0,
            **dict(weights or {}),
        }
        self.no_event_weight = no_event_weight

    def forward(
        self,
        events: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        events = _canonical_events(events)
        matches = self.matcher(events, targets)
        existence_target = torch.zeros_like(events["existence_logits"])
        components: Dict[str, torch.Tensor] = {}
        type_losses, boundary_losses, iou_losses = [], [], []
        attribute_losses, channel_losses = [], []
        confidence_losses = []
        confidence_target = None
        if self.weights.get("confidence", 0.0) > 0:
            if "confidence_logits" in events:
                confidence_target = torch.zeros_like(events["confidence_logits"])

        for batch_index, (source, destination) in enumerate(matches):
            if source.numel() == 0:
                continue
            existence_target[batch_index, source] = 1.0
            target = targets[batch_index]
            type_losses.append(
                F.cross_entropy(
                    events["type_logits"][batch_index, source],
                    target["types"].to(source.device)[destination].long(),
                )
            )
            predicted_boundaries = events["boundaries"][batch_index, source]
            target_boundaries = target["boundaries"].to(predicted_boundaries)[destination]
            boundary_losses.append(F.l1_loss(predicted_boundaries, target_boundaries))
            iou = interval_iou_1d(predicted_boundaries, target_boundaries)
            iou_losses.append(1.0 - iou.diagonal().mean())
            if "attributes" in events and "attributes" in target:
                predicted_attrs = events["attributes"][batch_index, source]
                target_attrs = target["attributes"].to(predicted_boundaries)[destination]
                attr_mask = target.get("attribute_mask")
                if attr_mask is not None:
                    mask = attr_mask.to(predicted_boundaries)[destination].bool()
                    if mask.any():
                        attribute_losses.append(
                            F.smooth_l1_loss(
                                predicted_attrs[mask],
                                target_attrs[mask],
                            )
                        )
                else:
                    attribute_losses.append(
                        F.smooth_l1_loss(predicted_attrs, target_attrs)
                    )
            if "channel_logits" in events and "channels" in target:
                channel_losses.append(
                    F.binary_cross_entropy_with_logits(
                        events["channel_logits"][batch_index, source],
                        target["channels"].to(predicted_boundaries)[destination].float(),
                    )
                )
            if confidence_target is not None and "confidence" in target:
                confidence_target[batch_index, source] = (
                    target["confidence"].to(predicted_boundaries)[destination].clamp(0.0, 1.0)
                )

        existence_weight = torch.where(
            existence_target.bool(),
            torch.ones_like(existence_target),
            torch.full_like(existence_target, self.no_event_weight),
        )
        components["exist"] = F.binary_cross_entropy_with_logits(
            events["existence_logits"], existence_target, weight=existence_weight
        )
        zero = events["existence_logits"].sum() * 0.0
        components["type"] = torch.stack(type_losses).mean() if type_losses else zero
        components["boundary"] = (
            torch.stack(boundary_losses).mean() if boundary_losses else zero
        )
        components["iou"] = torch.stack(iou_losses).mean() if iou_losses else zero
        components["attribute"] = (
            torch.stack(attribute_losses).mean() if attribute_losses else zero
        )
        components["channel"] = (
            torch.stack(channel_losses).mean() if channel_losses else zero
        )
        if confidence_target is not None:
            components["confidence"] = F.binary_cross_entropy_with_logits(
                events["confidence_logits"], confidence_target
            )
        else:
            components["confidence"] = zero
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components


def _canonical_events(events: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Accept both the documented logit schema and V1 decoder probabilities."""
    result = dict(events)
    eps = torch.finfo(next(iter(events.values())).dtype).eps

    def logits(value: torch.Tensor) -> torch.Tensor:
        if torch.all((value >= 0) & (value <= 1)):
            return torch.logit(value.clamp(eps, 1.0 - eps))
        return value

    if "existence_logits" not in result and "existence" in result:
        result["existence_logits"] = logits(result["existence"])
    if "type_logits" not in result and "type" in result:
        probability = result["type"].clamp_min(eps)
        result["type_logits"] = probability.log()
    if "boundaries" not in result and "start" in result and "end" in result:
        result["boundaries"] = torch.stack((result["start"], result["end"]), dim=-1)
    if "channel_logits" not in result and "channel_scores" in result:
        result["channel_logits"] = logits(result["channel_scores"])
    if "confidence_logits" not in result and "confidence" in result:
        result["confidence_logits"] = logits(result["confidence"])
    return result


def combined_loss(
    output: Mapping[str, Any],
    target: Mapping[str, Any],
    task: str,
    event_criterion: Optional[EventSetCriterion] = None,
    task_weight: float = 1.0,
    event_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses: Dict[str, torch.Tensor] = {}
    losses["task"] = task_loss(output["task_output"], target["task_target"], task)
    total = task_weight * losses["task"]
    if event_criterion is not None and target.get("events") is not None:
        event_total, event_parts = event_criterion(output["events"], target["events"])
        losses.update({f"event_{key}": value for key, value in event_parts.items()})
        total = total + event_weight * event_total
    losses["total"] = total
    return total, losses
