"""Metrics for morphology, HAR, anomaly detection, and forecasting."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target, prediction = np.asarray(target).ravel(), np.asarray(prediction).ravel()
    labels = np.union1d(target, prediction)
    f1_values, precisions, recalls, supports = [], [], [], []
    for label in labels:
        tp = np.sum((target == label) & (prediction == label))
        fp = np.sum((target != label) & (prediction == label))
        fn = np.sum((target == label) & (prediction != label))
        support = np.sum(target == label)
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
        f1_values.append(2 * tp / max(2 * tp + fp + fn, 1))
        supports.append(support)
    total = max(np.sum(supports), 1)
    confusion = np.asarray(
        [
            [
                np.sum((target == true_label) & (prediction == predicted_label))
                for predicted_label in labels
            ]
            for true_label in labels
        ],
        dtype=float,
    )
    correct = np.trace(confusion)
    predicted_count = confusion.sum(axis=0)
    true_count = confusion.sum(axis=1)
    denominator = np.sqrt(
        max(
            (confusion.sum() ** 2 - np.sum(predicted_count**2))
            * (confusion.sum() ** 2 - np.sum(true_count**2)),
            0.0,
        )
    )
    mcc = (
        (correct * confusion.sum() - np.dot(predicted_count, true_count))
        / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "accuracy": float(np.mean(target == prediction)) if target.size else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "weighted_f1": float(np.dot(f1_values, supports) / total),
        "macro_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "mcc": float(mcc),
    }


def anomaly_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    far_cap: Optional[float] = None,
) -> Dict[str, float]:
    target = np.asarray(target).astype(bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    def events(values: np.ndarray) -> list[tuple[int, int]]:
        changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))

    def event_stats(prediction_: np.ndarray) -> tuple[float, float, float, float, float]:
        truth_events, predicted_events = events(target), events(prediction_)
        matched_predictions: set[int] = set()
        detected, delays = 0, []
        ttd_values = []
        for start, end in truth_events:
            overlaps = [
                index
                for index, (pred_start, pred_end) in enumerate(predicted_events)
                if pred_start < end and pred_end > start
            ]
            if overlaps:
                detected += 1
                matched_predictions.update(overlaps)
                first = min(predicted_events[index][0] for index in overlaps)
                delay = max(0, first - start)
                delays.append(delay)
                ttd_values.append(delay)
            else:
                ttd_values.append(end - start)
        # Event-level precision: fraction of predicted events overlapping any GT
        # event (baseline event-prf definition). The old definition used the
        # matched-GT count, which exceeds 1.0 when several GT events share one
        # predicted span and is not comparable with the baseline.
        precision = len(matched_predictions) / max(len(predicted_events), 1)
        recall = detected / max(len(truth_events), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        delay = float(np.mean(delays)) if delays else float("nan")
        ttd = float(np.mean(ttd_values)) if ttd_values else float("nan")
        return precision, recall, f1, delay, ttd

    if threshold is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
        # Candidate thresholds ranked by event-F1; a far_cap (max false-alarm
        # rate) prunes degenerate thresholds that flag almost everything, which
        # otherwise collapse onto the lowest quantile on all-normal series.
        scored = []
        for value in candidates:
            prediction_ = scores >= value
            ef1 = event_stats(prediction_)[2]
            far = float(
                np.sum(prediction_ & ~target) / max(np.sum(~target), 1)
            )
            scored.append((float(ef1), far, float(value)))
        if far_cap is not None:
            acceptable = [item for item in scored if item[1] <= far_cap]
            if acceptable:
                best = max(acceptable, key=lambda item: (item[0], -item[1]))
            else:
                best = min(scored, key=lambda item: item[1])
        else:
            best = max(scored, key=lambda item: item[0])
        threshold = float(best[2])
    prediction = scores >= threshold
    tp = np.sum(target & prediction)
    fp = np.sum(~target & prediction)
    fn = np.sum(target & ~prediction)
    event_precision, event_recall, event_f1, delay, ttd = event_stats(prediction)
    normal = ~target
    far = float(np.sum(prediction & normal) / max(np.sum(normal), 1))
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "auroc": binary_auroc(target, scores),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "delay": delay,
        "ttd": ttd,
        "far": far,
        "threshold": float(threshold),
    }


def anomaly_metrics_from_windows(
    sample_ids: Sequence[str],
    labels_list: Sequence[np.ndarray],
    scores_list: Sequence[np.ndarray],
    *,
    threshold: Optional[float] = None,
    tune_threshold: bool = False,
    far_cap: Optional[float] = None,
    per_series_sweep: bool = False,
) -> Dict[str, float]:
    """Aggregate window metrics per series, then average across files.

    Threshold tuning uses the *concatenated* series (global) instead of the
    first series only, which previously let one all-normal series collapse the
    threshold to its lowest score quantile.  Metrics that are undefined on a
    series (e.g. auroc on an all-normal series) are averaged with nan-aware
    aggregation so a single degenerate series no longer poisons the result.

    When ``per_series_sweep`` is enabled (baseline TSB-AD protocol), every
    series tunes its own threshold to maximize event-F1 (subject to ``far_cap``)
    instead of sharing the global threshold; only event-level metrics are
    aggregated then, matching the baseline's per-file evaluation.
    """
    grouped: Dict[str, Dict[str, list]] = {}
    for sample_id, labels, scores in zip(sample_ids, labels_list, scores_list):
        key = str(sample_id).rsplit(":", 1)[0]
        bucket = grouped.setdefault(key, {"labels": [], "scores": []})
        bucket["labels"].append(np.asarray(labels).reshape(-1))
        bucket["scores"].append(np.asarray(scores, dtype=float).reshape(-1))

    if tune_threshold and threshold is None and not per_series_sweep:
        all_labels = np.concatenate([b["labels"] for b in grouped.values()])
        all_scores = np.concatenate([b["scores"] for b in grouped.values()])
        global_metrics = anomaly_metrics(
            all_labels, all_scores, threshold=None, far_cap=far_cap
        )
        threshold = float(global_metrics.get("threshold"))

    per_series = []
    for payload in grouped.values():
        labels = np.concatenate(payload["labels"])
        scores = np.concatenate(payload["scores"])
        metrics = anomaly_metrics(
            labels,
            scores,
            threshold=(
                None
                if per_series_sweep and tune_threshold
                else threshold
            ),
            far_cap=far_cap,
        )
        per_series.append(metrics)

    if not per_series:
        return {"auc_pr": float("nan"), "auroc": float("nan"), "n_series": 0.0}

    def _mean(values: Sequence[float]) -> float:
        finite = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    result = {
        key: _mean([item[key] for item in per_series if key in item])
        for key in per_series[0]
    }
    result["n_series"] = float(len(per_series))
    result["n_auroc_series"] = float(
        sum(1 for item in per_series if np.isfinite(item.get("auroc", float("nan"))))
    )
    if not per_series_sweep and threshold is not None:
        result["threshold"] = float(threshold)
    return result


def forecast_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    mse = float(np.mean(error**2))
    centered = target - np.mean(target)
    denominator = float(np.sum(centered**2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1e-8))),
        "r2": r2,
    }


def compute_task_metrics(
    task: str,
    target: np.ndarray,
    prediction: np.ndarray,
    scores: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, float]:
    if task in {"morphology", "event_classification"}:
        result = classification_metrics(target, prediction)
    elif task == "anomaly":
        result = anomaly_metrics(target, prediction if scores is None else scores)
    elif task == "forecast":
        result = forecast_metrics(target, prediction)
    else:
        raise ValueError(f"Unknown task: {task}")
    return {f"{prefix}{key}": value for key, value in result.items()}
