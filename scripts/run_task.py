#!/usr/bin/env python3
"""Unified JSON-emitting entry point for Stage 1 and downstream tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument(
        "--task",
        required=True,
        choices=[
            "stage0_base",
            "stage1_grounding",
            "controlled_scm",
            "stage1b_morphology",
            "morphology",
            "event_classification",
            "anomaly",
            "forecast",
            "forecast_diagnostics",
        ],
    )
    command.add_argument("--epochs", type=int, default=None)
    command.add_argument("--freeze-epochs", type=int, default=0)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--dataset")
    command.add_argument("--data-root", default="/data/panhongyu/TS-LLM/datasets")
    command.add_argument("--val-ratio", type=float, default=0.2)
    command.add_argument("--seq-len", type=int, default=96)
    command.add_argument("--pred-len", type=int, default=96)
    command.add_argument("--anomaly-window-size", type=int, default=512)
    command.add_argument(
        "--anomaly-far-cap",
        type=float,
        default=0.5,
        help="Max false-alarm rate for anomaly threshold tuning (prunes "
        "degenerate thresholds that flag almost everything)",
    )
    command.add_argument(
        "--balanced-anomaly",
        action="store_true",
        help="Balance normal/anomalous training windows for TSB-AD "
        "(weighted sampling so half of each batch is anomalous)",
    )
    command.add_argument(
        "--anomaly-per-series-sweep",
        action="store_true",
        help="TSB-AD eval: tune a per-series threshold maximizing event-F1 "
        "(baseline protocol) instead of a single global threshold",
    )
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--encoder-lr", type=float, default=3e-5)
    command.add_argument("--device", default="cuda")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--checkpoint")
    command.add_argument("--init-checkpoint")
    command.add_argument("--model-config", default="{}")
    command.add_argument("--smoke", action="store_true")
    command.add_argument("--measurement-consistency-weight", type=float, default=0.0)
    command.add_argument("--cardinality-weight", type=float, default=0.01)
    command.add_argument(
        "--width-prior-weight",
        type=float,
        default=None,
        help="P1 soft penalty for type-specific event width priors (default: WIDTH_PRIOR_WEIGHT env or 0)",
    )
    command.add_argument("--sparse-weight", type=float, default=0.01)
    command.add_argument("--entropy-weight", type=float, default=0.0)
    command.add_argument("--base-loss-weight", type=float, default=0.5)
    command.add_argument("--damage-weight", type=float, default=0.5)
    command.add_argument("--graph-norm-weight", type=float, default=0.01)
    command.add_argument(
        "--graph-advantage-weight",
        type=float,
        default=0.0,
        help=(
            "Reward the graph path for beating the base path by --damage-margin "
            "(0 disables). Complements --damage-weight, which only penalizes the "
            "graph being worse; without a positive term the residual can settle "
            "at 'no gain'."
        ),
    )
    command.add_argument("--damage-margin", type=float, default=0.0)
    command.add_argument("--gradient-clip", type=float, default=1.0)
    command.add_argument("--patience", type=int, default=5)
    command.add_argument("--min-delta", type=float, default=1e-4)
    command.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Diagnostic subset size for any Stage-2 task "
        "(balanced by class for morphology; first-N windows otherwise)",
    )
    command.add_argument(
        "--disable-graph-correction",
        action="store_true",
        help="Train only the task-native dense path; gate graph residuals to zero",
    )
    command.add_argument(
        "--forecast-loss-mode",
        choices=["mse_only", "full"],
        default="mse_only",
        help="Forecast loss: MSE only (default) or MSE+MAE+derivative+spectral",
    )
    command.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply inverse-frequency class weights (recommended for ECG5000)",
    )
    command.add_argument(
        "--selection-metric",
        default=None,
        help="Override validation checkpoint metric (e.g. macro_f1, mse)",
    )
    command.add_argument(
        "--training-stage",
        choices=["0", "1", "2", "3", "A", "B", "C", "D"],
        default="2",
        help="Curriculum stage: 0=base, 1=grounding, 2=Task0, 3=ranking (A-D legacy aliases)",
    )
    command.add_argument(
        "--detail-log",
        default=None,
        help="Append random sample dumps (events, relations, time series) to this file",
    )
    command.add_argument(
        "--detail-samples",
        type=int,
        default=None,
        help="Random samples to log per epoch (default: DETAIL_SAMPLES env or 2)",
    )
    command.add_argument(
        "--detail-max-points",
        type=int,
        default=None,
        help="Max time-series points per sample dump (default: DETAIL_MAX_POINTS env or 128)",
    )
    command.add_argument(
        "--detail-min-epoch",
        type=int,
        default=None,
        help="Start detail dumps from this epoch (default: DETAIL_MIN_EPOCH env or 1)",
    )
    command.add_argument(
        "--channel-loss-weight",
        type=float,
        default=0.5,
        help="Stage-1 event channel BCE weight in EventSetCriterion",
    )
    command.add_argument(
        "--event-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for rule-event auxiliary supervision in downstream tasks "
            "(morphology only; alignment is verified against official files)"
        ),
    )
    command.add_argument(
        "--oracle-events",
        action="store_true",
        help=(
            "Inject ground-truth rule events into the graph/task-head path "
            "(oracle_events ablation; Stage-2 tasks: morphology uses rule/aligned "
            "GT, HAR/anomaly/forecast use auto morphology rules). Also enabled by "
            "oracle_events=true in --model-config."
        ),
    )
    command.add_argument(
        "--oracle-train-ratio",
        type=float,
        default=1.0,
        help="Fraction of train batches that receive GT-event injection (0..1). Default 1.0 = full joint oracle.",
    )
    command.add_argument(
        "--oracle-curriculum-epochs",
        type=int,
        default=0,
        help="First N epochs inject GT events unconditionally; afterwards use --oracle-train-ratio.",
    )
    command.add_argument(
        "--oracle-eval-events",
        type=int,
        default=1,
        help="Inject GT events during eval (default 1). Set 0 to evaluate deployable metrics on real decoder events.",
    )
    command.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Transfer protocol: freeze encoder/event decoder/graph entirely; "
            "train only task_heads (never unfreezes)."
        ),
    )
    command.add_argument(
        "--label-fraction",
        type=float,
        default=1.0,
        help=(
            "Stratified fraction of the TRAIN split used for fine-tuning. "
            "Use 0.05/0.15/0.3/1.0 for few-shot transfer. "
            "Val/test size is controlled separately by --eval-fraction."
        ),
    )
    command.add_argument(
        "--eval-fraction",
        type=float,
        default=1.0,
        help=(
            "Stratified fraction of VAL and TEST used for per-epoch eval and "
            "final test (morphology / event_classification). "
            "Independent of --label-fraction; e.g. 0.3 speeds up eval."
        ),
    )
    command.add_argument(
        "--reset-task-head",
        action="store_true",
        help="Re-initialize task_heads after loading --init-checkpoint (cross-dataset transfer).",
    )
    return command


def _detail_settings(args: argparse.Namespace) -> Dict[str, Any]:
    detail_log = args.detail_log or os.environ.get("DETAIL_LOG")
    detail_samples = args.detail_samples
    if detail_samples is None:
        detail_samples = int(os.environ.get("DETAIL_SAMPLES", "2"))
    detail_max_points = args.detail_max_points
    if detail_max_points is None:
        detail_max_points = int(os.environ.get("DETAIL_MAX_POINTS", "128"))
    detail_min_epoch = args.detail_min_epoch
    if detail_min_epoch is None:
        detail_min_epoch = int(os.environ.get("DETAIL_MIN_EPOCH", "1"))
    return {
        "detail_log": detail_log or None,
        "detail_samples": detail_samples,
        "detail_max_points": detail_max_points,
        "detail_min_epoch": detail_min_epoch,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        model_kwargs = json.loads(args.model_config)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-config must be a JSON object") from exc
    if not isinstance(model_kwargs, dict):
        raise ValueError("--model-config must decode to a JSON object")
    defaults: Dict[str, Any] = {"max_channels": 64, "num_queries": 4}
    if args.task == "morphology":
        class_counts = {
            "CBF": 3,
            "ECG5000": 5,
            "FordA": 2,
            "FordB": 2,
            "ElectricDevices": 7,
            "StarLightCurves": 3,
            "SyntheticControl": 6,
            "TwoLeadECG": 2,
            "TwoPatterns": 4,
            "Wafer": 2,
            "Coffee": 2,
            "ItalyPowerDemand": 2,
            "Plane": 7,
            "Trace": 4,
            "MoteStrain": 2,
        }
        defaults["num_morphology_classes"] = class_counts.get(
            args.dataset or "SyntheticControl", 6
        )
    elif args.task == "event_classification":
        # SMD: binary window anomaly; UCI_HAR: 6 activities; UEA event sets
        # use official class counts (BasicMotions 4, UWaveGestureLibrary 8, ...).
        from src.data.uea_ucr import UEA_UCR_DATASETS, n_classes_for

        dataset = str(args.dataset or "UCI_HAR")
        if dataset.upper() == "SMD":
            defaults["num_event_classes"] = 2
        elif dataset in UEA_UCR_DATASETS:
            defaults["num_event_classes"] = n_classes_for(dataset)
        else:
            defaults["num_event_classes"] = 6
    elif args.task == "forecast":
        defaults["pred_len"] = 96
    defaults.update(model_kwargs)
    model_kwargs = defaults

    if args.task == "stage0_base":
        from src.training.pretrain_base import (
            BasePretrainConfig,
            build_stage0_loader,
            pretrain_base,
        )
        from src.training.train_task import build_model

        config = BasePretrainConfig(
            epochs=args.epochs or 10,
            lr=args.lr,
            encoder_lr=args.encoder_lr or args.lr,
            device=args.device,
            seed=args.seed,
            checkpoint=args.checkpoint or "checkpoints/stage0_base.pt",
            smoke=args.smoke,
            seq_len=args.seq_len,
            pred_len=min(args.pred_len, 16),
            max_channels=int(model_kwargs.get("max_channels", 64)),
            data_root=args.data_root,
        )
        model = build_model(**model_kwargs)
        if args.init_checkpoint:
            from src.training.train_task import _load_compatible_state

            _load_compatible_state(model, args.init_checkpoint)
        return pretrain_base(model, build_stage0_loader(config), config)

    if args.task in {"stage1_grounding", "controlled_scm", "stage1b_morphology"}:
        if args.task == "controlled_scm":
            from src.training.pretrain_events import (
                EventPretrainConfig,
                build_controlled_scm_loader,
                pretrain_events,
            )
            from src.training.train_task import build_model

            config = EventPretrainConfig(
                epochs=args.epochs or 10,
                lr=args.lr,
                device=args.device,
                seed=args.seed,
                checkpoint=args.checkpoint or "checkpoints/controlled_scm_stage1.pt",
                smoke=args.smoke,
                sparse_weight=args.sparse_weight,
                entropy_weight=args.entropy_weight,
                width_prior_weight=(
                    float(os.environ.get("WIDTH_PRIOR_WEIGHT", "0"))
                    if args.width_prior_weight is None
                    else args.width_prior_weight
                ),
                channel_loss_weight=args.channel_loss_weight,
                **_detail_settings(args),
            )
            model = build_model(**model_kwargs)
            if args.init_checkpoint:
                from src.training.train_task import _load_compatible_state

                _load_compatible_state(model, args.init_checkpoint)
            return pretrain_events(model, build_controlled_scm_loader(config), config)

        if args.task == "stage1b_morphology":
            from src.training.pretrain_events import (
                EventPretrainConfig,
                build_morphology_event_loader,
                pretrain_events,
            )
            from src.training.train_task import _load_compatible_state, build_model

            max_events = int(os.environ.get("MORPHOLOGY_MAX_EVENTS", model_kwargs.get("num_queries", 6)))
            model_kwargs["num_queries"] = max_events
            raw_datasets = os.environ.get("MORPHOLOGY_DATASETS")
            morphology_datasets = (
                [name for name in raw_datasets.split() if name] if raw_datasets else None
            )
            config = EventPretrainConfig(
                epochs=args.epochs or int(os.environ.get("STAGE1B_EPOCHS", "10")),
                lr=args.lr,
                device=args.device,
                seed=args.seed,
                checkpoint=args.checkpoint or "checkpoints/v1_stage1b_morphology.pt",
                smoke=args.smoke,
                sparse_weight=args.sparse_weight,
                entropy_weight=args.entropy_weight,
                width_prior_weight=(
                    float(os.environ.get("WIDTH_PRIOR_WEIGHT", "0"))
                    if args.width_prior_weight is None
                    else args.width_prior_weight
                ),
                channel_loss_weight=args.channel_loss_weight,
                max_events=max_events,
                morphology_datasets=morphology_datasets,
                use_raw_channel_only=os.environ.get("USE_RAW_CHANNEL_ONLY", "1") != "0",
                stage_name="morphology_events_stage1b",
                task_name="stage1b_morphology",
                **_detail_settings(args),
            )
            model = build_model(**model_kwargs)
            if args.init_checkpoint:
                _load_compatible_state(model, args.init_checkpoint)
            else:
                stage1 = os.environ.get(
                    "V1_STAGE1_CHECKPOINT", "checkpoints/v1_stage1_scm.pt"
                )
                if Path(stage1).is_file():
                    _load_compatible_state(model, stage1)
            return pretrain_events(model, build_morphology_event_loader(config), config)

        from src.training.pretrain_grounding import (
            GroundingPretrainConfig,
            build_grounding_pretrain_loader,
            pretrain_grounding,
        )
        from src.training.train_task import _load_compatible_state, build_model

        config = GroundingPretrainConfig(
            epochs=args.epochs or 15,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            checkpoint=args.checkpoint or "checkpoints/stage1_grounding.pt",
            smoke=args.smoke,
            seq_len=args.seq_len,
            pred_len=min(args.pred_len, 16),
            max_channels=int(model_kwargs.get("max_channels", 64)),
            data_root=args.data_root,
        )
        model = build_model(**model_kwargs)
        if args.init_checkpoint:
            _load_compatible_state(model, args.init_checkpoint)
        return pretrain_grounding(
            model, build_grounding_pretrain_loader(config), config
        )

    if args.task == "forecast_diagnostics":
        from src.training.train_task import TaskTrainConfig, run_forecast_diagnostics

        config = TaskTrainConfig(
            task="forecast",
            dataset=args.dataset or "ETTm1",
            data_root=args.data_root,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            init_checkpoint=args.init_checkpoint,
            smoke=args.smoke,
            overfit_samples=args.overfit_samples,
            max_channels=int(model_kwargs.get("max_channels", 64)),
        )
        return run_forecast_diagnostics(config, model_kwargs=model_kwargs)

    import dataclasses

    from src.training.train_task import TaskTrainConfig, train_task

    config_kwargs = {
        "task": args.task,
        "dataset": args.dataset,
        "epochs": args.epochs if args.epochs is not None else 20,
        "freeze_epochs": args.freeze_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "encoder_lr": args.encoder_lr,
        "device": args.device,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "init_checkpoint": args.init_checkpoint,
        "smoke": args.smoke,
        "measurement_consistency_weight": args.measurement_consistency_weight,
        "cardinality_weight": args.cardinality_weight,
        "width_prior_weight": (
            float(os.environ.get("WIDTH_PRIOR_WEIGHT", "0"))
            if args.width_prior_weight is None
            else args.width_prior_weight
        ),
        "sparse_weight": args.sparse_weight,
        "entropy_weight": args.entropy_weight,
        "base_loss_weight": args.base_loss_weight,
        "damage_weight": args.damage_weight,
        "graph_norm_weight": args.graph_norm_weight,
        "graph_advantage_weight": args.graph_advantage_weight,
        "damage_margin": args.damage_margin,
        "gradient_clip": args.gradient_clip,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "overfit_samples": args.overfit_samples,
        "max_channels": int(model_kwargs.get("max_channels", 64)),
        "data_root": args.data_root,
        "val_ratio": args.val_ratio,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "anomaly_window_size": args.anomaly_window_size,
        "anomaly_far_cap": args.anomaly_far_cap,
        "anomaly_per_series_sweep": args.anomaly_per_series_sweep,
        "balanced_anomaly": args.balanced_anomaly,
        "disable_graph_correction": args.disable_graph_correction,
        "event_aux_weight": args.event_aux_weight,
        "oracle_events": args.oracle_events
        or bool(model_kwargs.get("oracle_events", False)),
        "oracle_train_ratio": args.oracle_train_ratio,
        "oracle_curriculum_epochs": args.oracle_curriculum_epochs,
        "oracle_eval_events": bool(args.oracle_eval_events),
        "freeze_backbone": bool(getattr(args, "freeze_backbone", False)),
        "label_fraction": float(getattr(args, "label_fraction", 1.0)),
        "eval_fraction": float(getattr(args, "eval_fraction", 1.0)),
        "reset_task_head": bool(getattr(args, "reset_task_head", False)),
        **_detail_settings(args),
    }
    if args.selection_metric:
        config_kwargs["selection_metric"] = args.selection_metric
    allowed = {field.name for field in dataclasses.fields(TaskTrainConfig)}
    config = TaskTrainConfig(
        **{key: value for key, value in config_kwargs.items() if key in allowed}
    )
    return train_task(config, model_kwargs=model_kwargs)


def main() -> int:
    args = parser().parse_args()
    try:
        result = run(args)
    except Exception as exc:  # Keep automation output machine-readable.
        print(
            json.dumps(
                {"ok": False, "task": args.task, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 1
    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    print(json.dumps(json_safe({"ok": True, **result}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
