"""Bipartite matching for unordered event predictions."""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use SciPy when available and a deterministic fallback otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        rows, cols = cost.shape
        n = min(rows, cols)
        if n == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        # Exact for smoke tests/small query sets, greedy for larger matrices.
        if max(rows, cols) <= 8:
            best = None
            if rows <= cols:
                for candidate in itertools.permutations(range(cols), rows):
                    value = sum(cost[r, c] for r, c in enumerate(candidate))
                    if best is None or value < best[0]:
                        best = (value, np.arange(rows), np.asarray(candidate))
            else:
                for candidate in itertools.permutations(range(rows), cols):
                    value = sum(cost[r, c] for c, r in enumerate(candidate))
                    if best is None or value < best[0]:
                        best = (value, np.asarray(candidate), np.arange(cols))
            assert best is not None
            return best[1], best[2]
        remaining_rows, remaining_cols = set(range(rows)), set(range(cols))
        selected_rows, selected_cols = [], []
        while remaining_rows and remaining_cols:
            r, c = min(
                itertools.product(remaining_rows, remaining_cols),
                key=lambda pair: (cost[pair], pair),
            )
            selected_rows.append(r)
            selected_cols.append(c)
            remaining_rows.remove(r)
            remaining_cols.remove(c)
        return np.asarray(selected_rows), np.asarray(selected_cols)


def interval_iou_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between intervals ``a[..., 2]`` and ``b[..., 2]``."""
    left = torch.maximum(a[..., None, 0], b[..., 0])
    right = torch.minimum(a[..., None, 1], b[..., 1])
    intersection = (right - left).clamp_min(0)
    union = (
        (a[..., None, 1] - a[..., None, 0]).clamp_min(0)
        + (b[..., 1] - b[..., 0]).clamp_min(0)
        - intersection
    )
    return intersection / union.clamp_min(1e-8)


class HungarianMatcher:
    """Match predicted event slots to targets using a weighted joint cost."""

    def __init__(
        self,
        type_cost: float = 1.0,
        boundary_cost: float = 2.0,
        iou_cost: float = 2.0,
        attribute_cost: float = 0.5,
        channel_cost: float = 0.5,
    ) -> None:
        self.weights = {
            "type": type_cost,
            "boundary": boundary_cost,
            "iou": iou_cost,
            "attribute": attribute_cost,
            "channel": channel_cost,
        }

    @torch.no_grad()
    def __call__(
        self, events: Dict[str, torch.Tensor], targets: Sequence[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        type_values = events.get("type_logits", events.get("type"))
        if type_values is None:
            raise KeyError("events needs type_logits or type")
        type_prob = (
            type_values
            if torch.all((type_values >= 0) & (type_values <= 1))
            else type_values.softmax(-1)
        )
        boundaries = events.get("boundaries")
        if boundaries is None:
            if "start" not in events or "end" not in events:
                raise KeyError("events needs boundaries or start/end")
            boundaries = torch.stack((events["start"], events["end"]), dim=-1)
        matches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target in enumerate(targets):
            labels = target["types"].long()
            count = labels.numel()
            if count == 0:
                empty = torch.empty(0, dtype=torch.long, device=boundaries.device)
                matches.append((empty, empty))
                continue
            cost = -self.weights["type"] * type_prob[batch_index][:, labels]
            target_boundaries = target["boundaries"].to(boundaries)
            cost = cost + self.weights["boundary"] * torch.cdist(
                boundaries[batch_index], target_boundaries, p=1
            )
            cost = cost + self.weights["iou"] * (
                1.0 - interval_iou_1d(boundaries[batch_index], target_boundaries)
            )
            if "attributes" in target and "attributes" in events:
                cost = cost + self.weights["attribute"] * torch.cdist(
                    events["attributes"][batch_index],
                    target["attributes"].to(boundaries),
                    p=1,
                )
            channel_values = events.get("channel_logits", events.get("channel_scores"))
            if "channels" in target and channel_values is not None:
                target_channels = target["channels"].to(boundaries).float()
                prediction = channel_values[batch_index]
                if not torch.all((prediction >= 0) & (prediction <= 1)):
                    prediction = prediction.sigmoid()
                expanded_prediction = prediction[:, None, :].expand(-1, count, -1)
                expanded_target = target_channels[None, :, :].expand(
                    prediction.shape[0], -1, -1
                )
                cost = cost + self.weights["channel"] * F.binary_cross_entropy(
                    expanded_prediction, expanded_target, reduction="none"
                ).mean(-1)
            row, col = _linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=boundaries.device),
                    torch.as_tensor(col, dtype=torch.long, device=boundaries.device),
                )
            )
        return matches


match_events = HungarianMatcher
