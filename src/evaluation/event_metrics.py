"""Event detection and grounding metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


def iou_1d(first: np.ndarray, second: np.ndarray) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(0.0, first[1] - first[0]) + max(0.0, second[1] - second[0]) - intersection
    return float(intersection / max(union, 1e-12))


def event_metrics(
    predicted: Iterable[Mapping[str, object]],
    target: Iterable[Mapping[str, object]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    predicted, target = list(predicted), list(target)
    candidates = []
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(target):
            overlap = iou_1d(np.asarray(pred["boundary"]), np.asarray(truth["boundary"]))
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_index, target_index))
    matched_pred, matched_target, matches = set(), set(), []
    for overlap, pred_index, target_index in sorted(candidates, reverse=True):
        if pred_index not in matched_pred and target_index not in matched_target:
            matched_pred.add(pred_index)
            matched_target.add(target_index)
            matches.append((overlap, pred_index, target_index))
    tp, fp, fn = len(matches), len(predicted) - len(matches), len(target) - len(matches)
    type_correct = sum(
        predicted[pred_index].get("type") == target[target_index].get("type")
        for _, pred_index, target_index in matches
    )
    boundary_error = [
        np.abs(
            np.asarray(predicted[pred_index]["boundary"], dtype=float)
            - np.asarray(target[target_index]["boundary"], dtype=float)
        ).mean()
        for _, pred_index, target_index in matches
    ]
    return {
        "event_precision": float(tp / max(tp + fp, 1)),
        "event_recall": float(tp / max(tp + fn, 1)),
        "event_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "event_mean_iou": float(np.mean([item[0] for item in matches])) if matches else 0.0,
        "event_type_accuracy": float(type_correct / max(tp, 1)),
        "event_boundary_mae": float(np.mean(boundary_error)) if boundary_error else float("nan"),
    }
