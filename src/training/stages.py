"""Stage A–D training schedules for competitive hypothesis reasoning."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch


STAGE_DESCRIPTIONS = {
    "A": "task experts only; frozen shared encoder; no graph correction",
    "B": "evidence reader only; frozen task experts and encoder",
    "C": "gates only; frozen encoder; ranking + correction_helpful",
    "D": "limited joint fine-tune; encoder at 10–20x lower LR",
}


def _set_requires(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_backbone_train_heads(
    model: torch.nn.Module, task: str | None = None
) -> None:
    """Transfer protocol: freeze event extractor + graph path; train task heads only.

    Forward still runs the frozen event/graph stack so the head sees grounded
    event-graph features. When ``task`` is set, only that classification head
    (plus its optional ``*_graph`` twin) receives gradients.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    task_heads = getattr(model, "task_heads", None)
    if task_heads is None:
        model._training_stage = "transfer_head"
        return
    if task in {"morphology", "event_classification"}:
        for name, parameter in task_heads.named_parameters():
            if name.startswith(f"{task}.") or name.startswith(f"{task}_graph."):
                parameter.requires_grad = True
    else:
        _set_requires(task_heads, True)
    model._training_stage = "transfer_head"


def apply_training_stage(model: torch.nn.Module, stage: str) -> None:
    """Configure which modules train in each curriculum stage."""

    stage = stage.upper()
    if stage not in STAGE_DESCRIPTIONS:
        raise ValueError(f"Unknown training stage {stage!r}; expected one of A/B/C/D")

    encoder = getattr(model, "encoder", None)
    event_decoder = getattr(model, "event_decoder", None)
    graph_builder = getattr(model, "graph_builder", None)
    graph_reasoner = getattr(model, "graph_reasoner", None)
    graph_to_temporal = getattr(model, "graph_to_temporal", None)
    task_heads = getattr(model, "task_heads", None)
    event_field = getattr(model, "event_field_projection", None)
    reconstruction = getattr(model, "reconstruction_head", None)
    compat = getattr(model, "compat_scorer", None)

    train_all = [
        encoder,
        event_decoder,
        graph_builder,
        graph_reasoner,
        graph_to_temporal,
        task_heads,
        event_field,
        reconstruction,
        compat,
    ]
    for module in train_all:
        _set_requires(module, False)
    gate = getattr(model, "graph_gate", None)
    if isinstance(gate, torch.nn.Parameter):
        gate.requires_grad = False

    if stage == "A":
        _set_requires(task_heads, True)
        model._disable_graph_correction = True
        model._training_stage = "A"
        return

    if stage == "B":
        _set_requires(event_decoder, True)
        _set_requires(event_field, True)
        model._disable_graph_correction = True
        model._training_stage = "B"
        return

    if stage == "C":
        if task_heads is not None:
            for name, parameter in task_heads.named_parameters():
                if any(token in name for token in ("gates", "correction", "class_corrections")):
                    parameter.requires_grad = True
        model._disable_graph_correction = False
        model._training_stage = "C"
        return

    # Stage D: adapters + low-LR encoder
    _set_requires(task_heads, True)
    _set_requires(event_decoder, True)
    _set_requires(encoder, True)
    model._disable_graph_correction = False
    model._training_stage = "D"


def stage_encoder_lr_scale(stage: str) -> float:
    stage = stage.upper()
    if stage in {"A", "B", "C"}:
        return 0.0
    return 0.1


def gradient_diagnostics(
    model: torch.nn.Module, task_grads: Mapping[str, torch.Tensor]
) -> dict[str, float | dict[str, float]]:
    """Report gradient norms and pairwise cosine similarity across tasks."""

    norms = {task: float(tensor.norm().item()) for task, tensor in task_grads.items()}
    cosines: dict[str, float] = {}
    keys = sorted(task_grads)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            g_left, g_right = task_grads[left], task_grads[right]
            denom = g_left.norm() * g_right.norm()
            value = float((g_left @ g_right) / denom) if denom > 0 else float("nan")
            cosines[f"{left}/{right}"] = value
    return {"grad_norm": norms, "grad_cosine": cosines}


def collect_shared_grad_vector(model: torch.nn.Module) -> torch.Tensor | None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    chunks: list[torch.Tensor] = []
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            chunks.append(parameter.grad.detach().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks)
