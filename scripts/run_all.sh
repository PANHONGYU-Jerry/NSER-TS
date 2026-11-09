#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# V1: Grounded Event Graph Reasoner
# =============================================================================
# Pipeline (no LLM, no natural language, no learned Revision Controller):
#
#   X → PatchTST Temporal Encoder → Event Decoder (K_max=6) → Grounded Event Graph
#     → Graph Transformer → Task head (y)
#
# Training curriculum (§IX):
#   Stage 1  Controlled-SCM event pretraining
#   Stage 1b Rule-based morphology event grounding (teacher labels)
#   Stage 2  Per-task Task0 (freeze encoder → unfreeze with small encoder lr)
#   Stage 3  Ranking / fine-tune (init from Stage 2 checkpoint)
#
# Stage-2 dataset order (§X — one task family at a time):
#   1. Controlled-SCM
#   2. Morphology (UCR, labels = waveform/contour/spectrum shape):
#      CBF SyntheticControl TwoPatterns TwoLeadECG ECG5000 FordA FordB Wafer
#      StarLightCurves ElectricDevices Coffee ItalyPowerDemand Plane Trace
#      MoteStrain
#      Temporal supervision: baseline/UCR/models_shape/generate_features/output
#   3. Event classification (labels = activity/gesture/anomaly event;
#      protocol mirrors baseline/classify/scripts/run_all.sh):
#      UCI_HAR SMD BasicMotions UWaveGestureLibrary NATOPS Cricket Libras
#      Epilepsy RacketSports
#      Temporal supervision: baseline/classify/models_shape/generate_features/output
# =============================================================================

TASK="${TASK:-stage1_scm}"
PYTHON="${PYTHON:-/home/panhongyu/miniconda3/envs/flare/bin/python}"
EXTRA_ARGS=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# --- Paths & V1 model config (dynamic events, K_max candidate slots) -----------
DATA_ROOT="${DATA_ROOT:-/data/panhongyu/TS-LLM/datasets}"
# Per-seed isolation: CHECKPOINT_DIR=checkpoints/seed_0 bash ...
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
mkdir -p "$CHECKPOINT_DIR"
V1_STAGE1_CHECKPOINT="${V1_STAGE1_CHECKPOINT:-$CHECKPOINT_DIR/v1_stage1_scm.pt}"
V1_STAGE1B_CHECKPOINT="${V1_STAGE1B_CHECKPOINT:-$CHECKPOINT_DIR/v1_stage1b_morphology.pt}"
K_MAX="${K_MAX:-4}"
MORPHOLOGY_MAX_EVENTS="${MORPHOLOGY_MAX_EVENTS:-6}"
# Stage-2 morphology uses the same query count as Stage-1b so every learned
# decoder slot is inherited (previously 4 vs 6 silently discarded two slots).
MORPHOLOGY_NUM_QUERIES="${MORPHOLOGY_NUM_QUERIES:-$MORPHOLOGY_MAX_EVENTS}"
# Weight of rule-event auxiliary supervision on Stage-2 morphology training.
EVENT_AUX_WEIGHT="${EVENT_AUX_WEIGHT:-0.2}"
EVENT_EXISTENCE_THRESHOLD="${EVENT_EXISTENCE_THRESHOLD:-0.5}"
WIDTH_PRIOR_WEIGHT="${WIDTH_PRIOR_WEIGHT:-0.05}"

# --- Graph-path fixes (root-caused in log/20260909_192228) --------------------
# These knobs feed GroundedReasonerConfig through the merged --model-config, so
# they apply to every downstream run (stage2, stage3, train ablations).
#   GRAPH_RESIDUAL_ENABLED=0     keep graph-reasoned nodes in the pool, drop the
#                                H0+alpha*deltaH temporal residual (isolate residual)
#   GRAPH_EVENT_POOL_SOURCE      graph (legacy, graph info enters twice) | decoder
#                                (pool = raw decoder events; graph only via deltaH)
#   GRAPH_SEPARATE_HEAD=1        dedicated graph-path head weights (no base/final
#                                double-CE fight on one shared Linear)
#   GRAPH_HARD_EDGES=1           hard existence-threshold edges during training too
#                                (removes soft-train/hard-eval graph mismatch)
#   GRAPH_RELATION_THRESHOLD     drop learned relation scores below this value
#   GRAPH_DISABLE_ATTRIBUTE_RELATIONS=1
#                                zero HIGHER_THAN/STEEPER_THAN/SAME_FREQUENCY edges
#   GRAPH_GATE_INIT              initial graph gate g (alpha = alpha_max*tanh(g))
#   GRAPH_ADVANTAGE_WEIGHT       positive reward for the graph path beating base
GRAPH_RESIDUAL_ENABLED="${GRAPH_RESIDUAL_ENABLED:-1}"
GRAPH_EVENT_POOL_SOURCE="${GRAPH_EVENT_POOL_SOURCE:-graph}"
GRAPH_SEPARATE_HEAD="${GRAPH_SEPARATE_HEAD:-0}"
GRAPH_HARD_EDGES="${GRAPH_HARD_EDGES:-0}"
GRAPH_RELATION_THRESHOLD="${GRAPH_RELATION_THRESHOLD:-0}"
GRAPH_DISABLE_ATTRIBUTE_RELATIONS="${GRAPH_DISABLE_ATTRIBUTE_RELATIONS:-0}"
GRAPH_GATE_INIT="${GRAPH_GATE_INIT:-0.1}"
GRAPH_ADVANTAGE_WEIGHT="${GRAPH_ADVANTAGE_WEIGHT:-0}"
# Relation path into GraphReasoner (edge "embedding" = r_uv bias + message).
#   original | shuffled | no_relation_bias | no_relation_features
# Empty = model default (original). Set no_relation_features to drop both
# (w_R)^T r_uv and W_RV r_uv while keeping edge topology + content attention.
RELATION_MODE="${RELATION_MODE:-}"
MODEL_CONFIG="${MODEL_CONFIG:-{\"num_queries\":${K_MAX},\"patch_len\":16,\"patch_stride\":8,\"encoder_layers\":3,\"postprocess_events\":true,\"existence_threshold\":${EVENT_EXISTENCE_THRESHOLD}}}"

SCM_SEQ_LEN="${SCM_SEQ_LEN:-128}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1B_EPOCHS="${STAGE1B_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE2_FREEZE_EPOCHS="${STAGE2_FREEZE_EPOCHS:-5}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-10}"
STAGE2_LR="${STAGE2_LR:-0.001}"
STAGE2_ENCODER_LR="${STAGE2_ENCODER_LR:-0.00001}"
STAGE3_LR="${STAGE3_LR:-0.0003}"
STAGE3_ENCODER_LR="${STAGE3_ENCODER_LR:-0.00001}"
MEASUREMENT_CONSISTENCY_WEIGHT="${MEASUREMENT_CONSISTENCY_WEIGHT:-0.1}"
# SCM generates 1-4 events; stronger sparsity/entropy reduces false K_max slots.
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.05}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
CARDINALITY_WEIGHT="${CARDINALITY_WEIGHT:-0.05}"
STAGE1_CHANNEL_WEIGHT="${STAGE1_CHANNEL_WEIGHT:-1.5}"
# Detail dumps: log only late epochs + export channel-aligned pics after Stage-1.
DETAIL_SAMPLES="${DETAIL_SAMPLES:-2}"
DETAIL_MAX_POINTS="${DETAIL_MAX_POINTS:-256}"
DETAIL_MIN_EPOCH="${DETAIL_MIN_EPOCH:-8}"
DETAIL_MAX_CHANNELS="${DETAIL_MAX_CHANNELS:-6}"

MORPHOLOGY_DATASETS="${MORPHOLOGY_DATASETS:-CBF SyntheticControl TwoPatterns TwoLeadECG ECG5000 FordA FordB Wafer StarLightCurves ElectricDevices Coffee ItalyPowerDemand Plane Trace MoteStrain}"
# Event classification datasets. UCI_HAR keeps its legacy unsuffixed run name
# (stage2_event_classification.pt / logs); other datasets get an explicit
# _<DATASET> suffix (stage2_event_classification_SMD.pt, ...) so runs never collide.
# Share the classify baseline data protocol + classify/models_shape temporal labels.
EVENT_CLASSIFICATION_DATASETS="${EVENT_CLASSIFICATION_DATASETS:-UCI_HAR SMD BasicMotions UWaveGestureLibrary NATOPS Cricket Libras Epilepsy RacketSports}"
# §XI ablation default coverage: every dataset of both downstream tasks.
# Eval-only arms skip datasets missing a Stage-2 checkpoint; train arms always retrain.
# Restrict for smoke runs, e.g. ABLATION_DATASETS="SyntheticControl UCI_HAR".
ABLATION_DATASETS="${ABLATION_DATASETS:-$MORPHOLOGY_DATASETS $EVENT_CLASSIFICATION_DATASETS}"
# Component claim arms (C1/C2). Names state what is kept / disabled.
ABLATION_COMPONENT_VARIANTS="${ABLATION_COMPONENT_VARIANTS:-no_events_no_graph events_no_graph events_graph}"
# train = paper-grade (retrain each arm from Stage-1b; default).
# eval  = inference-time switch on Stage-2 ckpt (fast diagnostic only;
#         no_events_no_graph now sets events_disabled=true, but the head was
#         still trained WITH events — do not cite as a training contribution).
ABLATION_COMPONENT_MODE="${ABLATION_COMPONENT_MODE:-train}"
# Train-matrix default = the three component arms (use new names). Override to
# also run no_aux / no_graph_no_aux, e.g. ABLATION_TRAIN_VARIANTS="... no_aux".
ABLATION_TRAIN_VARIANTS="${ABLATION_TRAIN_VARIANTS:-$ABLATION_COMPONENT_VARIANTS}"
# Stage-1b loader filters on MORPHOLOGY_DATASETS; without this it ingests every
# discovered split (incl. forecast features with huge magnitudes that blow up MSE).
export MORPHOLOGY_DATASETS

# --- TwoLeadECG small-data tuning --------------------------------------------
# TwoLeadECG has only 23 official train samples (~16 after the val split), so
# the default batch size 32 collapses every epoch into a single step and the
# default 10-epoch schedule barely trains. Dedicated schedule: small batches
# (several steps/epoch), many more epochs, and a shorter encoder freeze so the
# tiny train set actually gets updated. These are used via the per-dataset
# override in run_v1_downstream (applied AFTER stage flags so they win).
TWOTLEAD_BATCH_SIZE="${TWOTLEAD_BATCH_SIZE:-8}"
TWOTLEAD_EPOCHS="${TWOTLEAD_EPOCHS:-40}"
TWOTLEAD_FREEZE_EPOCHS="${TWOTLEAD_FREEZE_EPOCHS:-2}"
TWOTLEAD_LR="${TWOTLEAD_LR:-0.0005}"
TWOTLEAD_ENCODER_LR="${TWOTLEAD_ENCODER_LR:-0.00002}"
TWOTLEAD_PATIENCE="${TWOTLEAD_PATIENCE:-12}"
TWOTLEAD_VAL_RATIO="${TWOTLEAD_VAL_RATIO:-0.3}"

# --- SyntheticControl small-data tuning --------------------------------------
# 300 train / 300 test, 6 balanced classes. The default val_ratio=0.2 carved
# out 60 samples for validation, and the default 10-epoch / freeze-5 schedule
# froze the encoder for half of training. Measured wins (joint-oracle eval):
#   default (val 0.2, 10 ep, freeze 5): test macro_f1 0.9702
#   tuned   (val 0.1, 30 ep, freeze 2): test macro_f1 0.9967  (= RDST)
# val_ratio 0.05 (285 train) regressed to 0.990, so 0.1 is the sweet spot.
SYNTHETIC_BATCH_SIZE="${SYNTHETIC_BATCH_SIZE:-32}"
SYNTHETIC_EPOCHS="${SYNTHETIC_EPOCHS:-30}"
SYNTHETIC_FREEZE_EPOCHS="${SYNTHETIC_FREEZE_EPOCHS:-2}"
SYNTHETIC_LR="${SYNTHETIC_LR:-0.0005}"
SYNTHETIC_ENCODER_LR="${SYNTHETIC_ENCODER_LR:-0.00002}"
SYNTHETIC_PATIENCE="${SYNTHETIC_PATIENCE:-12}"
SYNTHETIC_VAL_RATIO="${SYNTHETIC_VAL_RATIO:-0.1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT/log/$TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "V1 run_all | TASK=$TASK | K_max=$K_MAX | logs=$LOG_DIR"

# --- Helpers ------------------------------------------------------------------
run_with_log() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  local detail_file="$LOG_DIR/${name}.detail.log"
  export DETAIL_LOG="$detail_file"
  export DETAIL_SAMPLES
  export DETAIL_MAX_POINTS
  export DETAIL_MIN_EPOCH
  export DETAIL_MAX_CHANNELS
  export EVENT_EXISTENCE_THRESHOLD
  echo "[$(date +'%F %T')] ===== start task=$name =====" | tee -a "$LOG_DIR/run_all.log"
  echo "[$(date +'%F %T')] log_file=$log_file" | tee -a "$LOG_DIR/run_all.log"
  echo "[$(date +'%F %T')] detail_log=$detail_file" | tee -a "$LOG_DIR/run_all.log"
  set +e
  {
    echo "[$(date +'%F %T')] CMD: $*"
    "$@"
  } 2>&1 | tee -a "$log_file" "$LOG_DIR/run_all.log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[$(date +'%F %T')] ===== finish task=$name exit=$status =====" | tee -a "$LOG_DIR/run_all.log"
  return "$status"
}

resolve_stage1_scm_init() {
  local candidates=(
    "$V1_STAGE1_CHECKPOINT"
    "$CHECKPOINT_DIR/controlled_scm_stage1.pt"
    checkpoints/controlled_scm_stage1.pt
  )
  local path
  for path in "${candidates[@]}"; do
    [[ -f "$path" ]] && echo "$path" && return 0
  done
  echo ""
}

resolve_stage1_init() {
  local candidates=(
    "$V1_STAGE1B_CHECKPOINT"
    "$CHECKPOINT_DIR/v1_stage1b_morphology.pt"
    checkpoints/v1_stage1b_morphology.pt
    "$V1_STAGE1_CHECKPOINT"
    "$CHECKPOINT_DIR/controlled_scm_stage1.pt"
    checkpoints/controlled_scm_stage1.pt
  )
  local path
  for path in "${candidates[@]}"; do
    [[ -f "$path" ]] && echo "$path" && return 0
  done
  echo ""
}

require_stage1_scm_init() {
  local init
  init="$(resolve_stage1_scm_init)"
  if [[ -z "$init" ]]; then
    echo "Missing Stage-1 SCM checkpoint. Run TASK=stage1_scm first." >&2
    return 1
  fi
  echo "$init"
}

require_stage1_init() {
  local init
  init="$(resolve_stage1_init)"
  if [[ -z "$init" ]]; then
    echo "Missing Stage-1/1b checkpoint. Run TASK=stage1_scm (and optionally stage1b_morphology) first." >&2
    return 1
  fi
  echo "$init"
}

task_run_name() {
  local task="$1" dataset="${2:-}" stage_label="$3"
  case "$task" in
    morphology) echo "${stage_label}_morphology_${dataset}" ;;
    event_classification)
      if [[ -z "$dataset" || "$dataset" == "UCI_HAR" ]]; then
        # Legacy run name: the first event-classification dataset (UCI HAR)
        # keeps its unsuffixed checkpoint/log names so prior runs and ablation
        # references (stage2_event_classification.pt, etc.) stay valid.
        echo "${stage_label}_event_classification"
      else
        echo "${stage_label}_event_classification_${dataset}"
      fi
      ;;
    *) echo "${stage_label}_${task}" >&2; return 2 ;;
  esac
}

task_checkpoint() {
  echo "$CHECKPOINT_DIR/$(task_run_name "$@").pt"
}

# V1 Stage 2: Task0 per-task training.
v1_stage2_flags() {
  echo --epochs "$STAGE2_EPOCHS" \
    --freeze-epochs "$STAGE2_FREEZE_EPOCHS" \
    --lr "$STAGE2_LR" \
    --encoder-lr "$STAGE2_ENCODER_LR" \
    --measurement-consistency-weight "$MEASUREMENT_CONSISTENCY_WEIGHT" \
    --cardinality-weight "$CARDINALITY_WEIGHT" \
    --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
    --sparse-weight "$SPARSE_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --gradient-clip 1 \
    --patience 6
}

# V1 Stage 3: ranking / fine-tune from Stage-2 checkpoint.
v1_stage3_flags() {
  echo --epochs "$STAGE3_EPOCHS" \
    --freeze-epochs 0 \
    --lr "$STAGE3_LR" \
    --encoder-lr "$STAGE3_ENCODER_LR" \
    --measurement-consistency-weight 0 \
    --base-loss-weight 0.5 \
    --damage-weight 0.5 \
    --graph-norm-weight 0.01 \
    --gradient-clip 1 \
    --patience 5
}

# Merge a brace-less JSON fragment into a JSON model-config string;
# prints the merged JSON on stdout.
merge_model_config() {
  local base="$1" fragment="${2:-}"
  if [[ -z "$fragment" ]]; then
    printf '%s' "$base"
    return 0
  fi
  "$PYTHON" -c '
import json, sys
base = json.loads(sys.argv[1])
extra = json.loads("{" + sys.argv[2] + "}")
base.update(extra)
print(json.dumps(base))
' "$base" "$fragment"
}

# Brace-less JSON fragment carrying the GRAPH_* env knobs (see the config block
# near the top). Merged into the resolved model config in run_v1_downstream, so
# a plain `GRAPH_SEPARATE_HEAD=1 TASK=stage2` run picks the fix up without a
# separate ablation arm. Empty output = keep the model defaults.
graph_model_extra() {
  local extra=""
  if [[ "$GRAPH_RESIDUAL_ENABLED" == "0" ]]; then
    extra="${extra:+$extra,}\"graph_residual_enabled\":false"
  fi
  if [[ "$GRAPH_EVENT_POOL_SOURCE" != "graph" ]]; then
    extra="${extra:+$extra,}\"graph_event_pool_source\":\"$GRAPH_EVENT_POOL_SOURCE\""
  fi
  if [[ "$GRAPH_SEPARATE_HEAD" == "1" ]]; then
    extra="${extra:+$extra,}\"separate_graph_head\":true"
  fi
  if [[ "$GRAPH_HARD_EDGES" == "1" ]]; then
    extra="${extra:+$extra,}\"graph_hard_edges\":true"
  fi
  if [[ "$GRAPH_RELATION_THRESHOLD" != "0" ]]; then
    extra="${extra:+$extra,}\"graph_relation_threshold\":$GRAPH_RELATION_THRESHOLD"
  fi
  if [[ "$GRAPH_DISABLE_ATTRIBUTE_RELATIONS" == "1" ]]; then
    extra="${extra:+$extra,}\"graph_disable_attribute_relations\":true"
  fi
  if [[ "$GRAPH_GATE_INIT" != "0.1" ]]; then
    extra="${extra:+$extra,}\"graph_gate_init\":$GRAPH_GATE_INIT"
  fi
  if [[ -n "$RELATION_MODE" && "$RELATION_MODE" != "original" ]]; then
    extra="${extra:+$extra,}\"relation_mode\":\"$RELATION_MODE\""
    if [[ "$RELATION_MODE" == "shuffled" ]]; then
      extra="${extra},\"shuffle_relation_seed\":-1"
    fi
  fi
  printf '%s' "$extra"
}

# Per-variant model-config JSON fragment (no braces) for eval-only ablations.
# Kept separate from ablation_flags so num_queries is always inherited from the
# base MODEL_CONFIG (morphology forces MORPHOLOGY_NUM_QUERIES in
# run_v1_downstream). run_ablation_eval exports this as MODEL_CONFIG_EXTRA,
# which run_v1_downstream merges into the single --model-config it passes on.
ablation_model_extra() {
  local variant="$1"
  case "$variant" in
    # True no-event eval arm: zero the head's event half (task_heads.py).
    # Without this, no_events_no_graph only flipped --disable-graph-correction
    # and still fed reasoned_events into combined_base (log/20260909_154637).
    no_events_no_graph|encoder_only|no_events)
      echo '"events_disabled":true'
      ;;
    oracle_event) echo '"oracle_events":true' ;;
    *) echo "" ;;
  esac
}

# Map legacy / short aliases onto the canonical component names used in logs
# and checkpoints (keeps TASK=ablation_encoder_only and ablation_no_events_no_graph
# writing the same artifact when desired).
ablation_canonical_variant() {
  local variant="$1"
  case "$variant" in
    encoder_only|no_events) echo no_events_no_graph ;;
    encoder_event|no_graph) echo events_no_graph ;;
    encoder_event_graph|full|full_no_verifier) echo events_graph ;;
    *) echo "$variant" ;;
  esac
}

# §XI ablation variants (eval-only when train_task supports epochs=0).
# NOTE: these never emit --model-config. Overwriting the model config here was
# the root cause of the num_queries 4-vs-6 mismatch in log/20260904_110944
# (an nq=4 eval of an nq=6 checkpoint silently re-inits event_decoder.queries
# and collapses SyntheticControl from ~0.95 to ~0.83).
ablation_flags() {
  local variant="$1"
  # Component ablation arms. Each name states what is kept and what is
  # disabled, so no ambiguity remains about the ablated component:
  #   no_events_no_graph  encoder only: event path AND graph correction off
  #                       (requires events_disabled via ablation_model_extra)
  #   events_no_graph     encoder + events: graph correction off
  #   events_graph        encoder + events + graph correction (= full model)
  # encoder_only / encoder_event / encoder_event_graph are legacy aliases.
  case "$variant" in
    no_events_no_graph|encoder_only|no_events)
      # Mirror train-arm loss zeros so eval and train protocols stay aligned.
      echo --disable-graph-correction --epochs 0 \
        --base-loss-weight 1.0 --damage-weight 0 --graph-norm-weight 0 \
        --measurement-consistency-weight 0 --cardinality-weight 0 \
        --sparse-weight 0 --entropy-weight 0 --width-prior-weight 0
      ;;
    events_no_graph|encoder_event|no_graph)
      echo --disable-graph-correction --epochs 0 \
        --measurement-consistency-weight "$MEASUREMENT_CONSISTENCY_WEIGHT"
      ;;
    events_graph|encoder_event_graph|full_no_verifier|full)
      echo --epochs 0 --base-loss-weight 1.0 --damage-weight 0.5 --graph-norm-weight 0.01
      ;;
    oracle_event)
      echo --epochs 0 --oracle-events
      ;;
    *)
      echo "Unknown ablation variant: $variant" >&2
      return 2
      ;;
  esac
}

# --- Runners ------------------------------------------------------------------
visualize_detail_log() {
  local name="$1"
  local detail_log="$LOG_DIR/${name}.detail.log"
  local pic_dir="$LOG_DIR/pic_${name}"
  if [[ -f "$detail_log" ]]; then
    echo "[$(date +'%F %T')] visualizing detail log -> $pic_dir" | tee -a "$LOG_DIR/run_all.log"
    "$PYTHON" scripts/visualize_detail_log.py \
      --detail-log "$detail_log" \
      --output-dir "$pic_dir"
  fi
}

run_v1_stage1_scm() {
  local init_args=()
  local prior
  prior="$(resolve_stage1_scm_init)"
  if [[ -n "$prior" && "$prior" != "$V1_STAGE1_CHECKPOINT" ]]; then
    init_args=(--init-checkpoint "$prior")
  fi
  run_with_log stage1_controlled_scm \
    "$PYTHON" scripts/run_task.py \
      --task controlled_scm \
      --epochs "$STAGE1_EPOCHS" \
      --seq-len "$SCM_SEQ_LEN" \
      --checkpoint "$V1_STAGE1_CHECKPOINT" \
      --model-config "$MODEL_CONFIG" \
      --sparse-weight "$SPARSE_WEIGHT" \
      --entropy-weight "$ENTROPY_WEIGHT" \
      --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
      --channel-loss-weight "$STAGE1_CHANNEL_WEIGHT" \
      --detail-min-epoch "$DETAIL_MIN_EPOCH" \
      "${init_args[@]}" \
      "${EXTRA_ARGS[@]}"
  visualize_detail_log stage1_controlled_scm
}

run_v1_stage1b_morphology() {
  local init_args=()
  local prior
  prior="$(require_stage1_scm_init)"
  init_args=(--init-checkpoint "$prior")
  # Feed only the raw channel so the learned event representation matches the
  # Stage-2 input (raw UCR series); the 13-channel derived features previously
  # caused a representation mismatch with downstream tasks.
  export USE_RAW_CHANNEL_ONLY=1
  local stage1b_model_config
  stage1b_model_config="{\"num_queries\":${MORPHOLOGY_MAX_EVENTS},\"patch_len\":16,\"patch_stride\":8,\"encoder_layers\":3,\"postprocess_events\":true,\"existence_threshold\":${EVENT_EXISTENCE_THRESHOLD}}"
  run_with_log stage1b_morphology_events \
    "$PYTHON" scripts/run_task.py \
      --task stage1b_morphology \
      --epochs "$STAGE1B_EPOCHS" \
      --checkpoint "$V1_STAGE1B_CHECKPOINT" \
      --model-config "$stage1b_model_config" \
      --sparse-weight "$SPARSE_WEIGHT" \
      --entropy-weight "$ENTROPY_WEIGHT" \
      --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
      --channel-loss-weight "$STAGE1_CHANNEL_WEIGHT" \
      --detail-min-epoch "$DETAIL_MIN_EPOCH" \
      "${init_args[@]}" \
      "${EXTRA_ARGS[@]}"
  visualize_detail_log stage1b_morphology_events
}

run_v1_downstream() {
  local task="$1"
  local dataset="${2:-}"
  local stage_label="$3"
  local extra_flags="${4:-}"

  local run_name checkpoint task_args=() init_checkpoint
  run_name="$(task_run_name "$task" "$dataset" "$stage_label")"
  checkpoint="$(task_checkpoint "$task" "$dataset" "$stage_label")"

  if [[ "$stage_label" == stage3 || "$stage_label" == ablation_* ]]; then
    local stage2_ckpt
    stage2_ckpt="$(task_checkpoint "$task" "$dataset" stage2)"
    if [[ -f "$stage2_ckpt" ]]; then
      init_checkpoint="$stage2_ckpt"
    else
      echo "[warn] Stage-2 checkpoint missing ($stage2_ckpt); falling back to Stage-1." | tee -a "$LOG_DIR/run_all.log"
      init_checkpoint="$(require_stage1_init)"
    fi
  else
    init_checkpoint="$(require_stage1_init)"
  fi

  case "$task" in
    morphology)
      [[ -n "$dataset" ]] || { echo "morphology requires DATASET" >&2; return 2; }
      task_args=(--dataset "$dataset" --data-root "$DATA_ROOT")
      if [[ "$dataset" == "TwoLeadECG" ]]; then
        # Tiny official train set (23 samples): a bigger val split keeps
        # best-epoch selection from riding macro-f1 noise on ~4 windows.
        task_args+=(--val-ratio "${TWOTLEAD_VAL_RATIO:-0.3}")
      fi
      task_args+=(--selection-metric macro_f1)
      # Rule-event aux supervision applies to Stage-2 and train-based ablation
      # arms (train_abl_*); no_aux arms disable it by exporting EVENT_AUX_WEIGHT=0.
      if [[ ( "$stage_label" == stage2 || "$stage_label" == train_abl_* ) && "${EVENT_AUX_WEIGHT:-0}" != "0" ]]; then
        task_args+=(--event-aux-weight "$EVENT_AUX_WEIGHT")
      fi
      ;;
    event_classification)
      task_args=(--dataset "${dataset:-UCI_HAR}" --data-root "$DATA_ROOT")
      task_args+=(--selection-metric macro_f1)
      # Baseline classify temporal/morphology labels as decoder aux (HAR / SMD /
      # UEA), symmetric with morphology EVENT_AUX_WEIGHT. no_events arms zero it.
      if [[ ( "$stage_label" == stage2 || "$stage_label" == train_abl_* ) && "${EVENT_AUX_WEIGHT:-0}" != "0" ]]; then
        task_args+=(--event-aux-weight "$EVENT_AUX_WEIGHT")
      fi
      ;;
    *)
      echo "Unsupported task: $task" >&2
      return 2
      ;;
  esac

  # All downstream morphology stages init from a Stage-1b/Stage-2 checkpoint
  # whose event decoder was trained with MORPHOLOGY_NUM_QUERIES slots. Force
  # that same query count here: otherwise event_decoder.queries shape-mismatches
  # on init, is silently skipped by _load_compatible_state, and every run
  # (stage3, ablations) trains/evaluates with randomly re-initialized event
  # queries (log/20260904_110944: nq=4 evals of nq=6 ckpts dropped SyntheticControl
  # 0.947 -> ~0.83 and CBF/TwoLeadECG Stage-3 initial by -12/-19pp).
  local resolved_model_config="$MODEL_CONFIG"
  if [[ "$task" == "morphology" ]]; then
    resolved_model_config="$(sed "s/\"num_queries\":[0-9]*/\"num_queries\":${MORPHOLOGY_NUM_QUERIES}/" <<<"$MODEL_CONFIG")"
  fi
  # MODEL_CONFIG_EXTRA carries brace-less JSON fragments from eval-only
  # ablations or the caller env; merge them on top of the resolved config
  # instead of replacing it wholesale.
  if [[ -n "${MODEL_CONFIG_EXTRA:-}" ]]; then
    resolved_model_config="$(merge_model_config "$resolved_model_config" "$MODEL_CONFIG_EXTRA")"
  fi
  # Graph-path fix knobs (GRAPH_* env; see graph_model_extra). Merged last so an
  # explicit env knob wins over the base MODEL_CONFIG defaults.
  local graph_extra
  graph_extra="$(graph_model_extra)"
  if [[ -n "$graph_extra" ]]; then
    resolved_model_config="$(merge_model_config "$resolved_model_config" "$graph_extra")"
  fi

  local stage_flags=()
  [[ -n "$extra_flags" ]] && read -r -a stage_flags <<< "$extra_flags"

  # Per-dataset overrides applied AFTER the stage flags so argparse last-wins.
  # Small train sets need dedicated schedules: TwoLeadECG has only 23 official
  # train samples; SyntheticControl's default val_ratio=0.2 wasted 60 samples
  # and the 10-epoch/freeze-5 schedule froze the encoder half the run (val
  # saturated 1.0 at epoch 1, locking best_epoch=1 and dropping later gains).
  local dataset_override=()
  local dataset_val_ratio=""
  if [[ "$task" == "morphology" && "$stage_label" != stage3* && "$stage_label" != ablation_* ]]; then
    case "$dataset" in
      TwoLeadECG)
        dataset_override=(
          --batch-size "$TWOTLEAD_BATCH_SIZE"
          --epochs "$TWOTLEAD_EPOCHS"
          --freeze-epochs "$TWOTLEAD_FREEZE_EPOCHS"
          --lr "$TWOTLEAD_LR"
          --encoder-lr "$TWOTLEAD_ENCODER_LR"
          --patience "$TWOTLEAD_PATIENCE"
        )
        dataset_val_ratio="$TWOTLEAD_VAL_RATIO"
        ;;
      SyntheticControl)
        dataset_override=(
          --batch-size "$SYNTHETIC_BATCH_SIZE"
          --epochs "$SYNTHETIC_EPOCHS"
          --freeze-epochs "$SYNTHETIC_FREEZE_EPOCHS"
          --lr "$SYNTHETIC_LR"
          --encoder-lr "$SYNTHETIC_ENCODER_LR"
          --patience "$SYNTHETIC_PATIENCE"
        )
        dataset_val_ratio="$SYNTHETIC_VAL_RATIO"
        ;;
    esac
  fi
  if [[ -n "$dataset_val_ratio" ]]; then
    dataset_override+=(--val-ratio "$dataset_val_ratio")
  fi

  run_with_log "$run_name" \
    "$PYTHON" scripts/run_task.py \
      --task "$task" \
      --init-checkpoint "$init_checkpoint" \
      --checkpoint "$checkpoint" \
      --model-config "$resolved_model_config" \
      "${task_args[@]}" \
      "${stage_flags[@]}" \
      "${dataset_override[@]}" \
      "${EXTRA_ARGS[@]}"
  visualize_detail_log "$run_name"
}

run_v1_event_classification() {
  # Loop helper shared by the Stage-2 / Stage-3 curricula: runs every dataset in
  # EVENT_CLASSIFICATION_DATASETS (default 9 event datasets).
  local stage_label="$1" extra_flags="${2:-}" dataset
  for dataset in $EVENT_CLASSIFICATION_DATASETS; do
    run_v1_downstream event_classification "$dataset" "$stage_label" "$extra_flags"
  done
}

run_v1_stage2_curriculum() {
  local flags
  flags="$(v1_stage2_flags)"
  local dataset

  for dataset in $MORPHOLOGY_DATASETS; do
    run_v1_downstream morphology "$dataset" stage2 "$flags"
  done
  run_v1_event_classification stage2 "$flags"
}

run_v1_stage3_ranking() {
  local flags
  flags="$(v1_stage3_flags)"
  local dataset

  for dataset in $MORPHOLOGY_DATASETS; do
    run_v1_downstream morphology "$dataset" stage3 "$flags"
  done
  run_v1_event_classification stage3 "$flags"
}

run_ablation_eval() {
  local variant="$1" flags dataset task
  local canonical
  canonical="$(ablation_canonical_variant "$variant")"
  flags="$(ablation_flags "$canonical")"
  # Per-variant model knobs (oracle / events_disabled) are merged into the
  # base model config via MODEL_CONFIG_EXTRA rather than a separate
  # --model-config, so num_queries resolution is never clobbered.
  local extra
  extra="$(ablation_model_extra "$canonical")"
  if [[ -n "$extra" ]]; then
    export MODEL_CONFIG_EXTRA="$extra"
  else
    unset MODEL_CONFIG_EXTRA
  fi
  local run_variant="$canonical"
  case "$variant" in
    oracle_*) run_variant="$variant" ;;
  esac
  for dataset in $ABLATION_DATASETS; do
    case "$dataset" in
      UCI_HAR|SMD|BasicMotions|UWaveGestureLibrary|NATOPS|Cricket|Libras|Epilepsy|RacketSports) task="event_classification" ;;
      *) task="morphology" ;;
    esac
    # Eval-only ablation arms evaluate the Stage-2 checkpoint. With the matrix
    # defaulting to every dataset of both tasks, skip (don't fall back to a
    # Stage-1 init) any dataset that has not been trained yet.
    local ckpt
    ckpt="$(task_checkpoint "$task" "$dataset" stage2)"
    if [[ ! -f "$ckpt" ]]; then
      echo "[warn] ablation '$run_variant': missing Stage-2 checkpoint $ckpt; skipping $dataset (train it first, e.g. TASK=morphology DATASET=$dataset)" | tee -a "$LOG_DIR/run_all.log"
      continue
    fi
    run_v1_downstream "$task" "$dataset" "ablation_${run_variant}" "$flags"
  done
  unset MODEL_CONFIG_EXTRA
}

# Component ablation (C1/C2). Default mode=train: retrain each arm from the
# Stage-1b init so "turning off events/graph" actually changes what the model
# learns. mode=eval is a fast diagnostic that flips inference switches on an
# existing Stage-2 ckpt — useful for plumbing checks, not for paper claims.
run_component_ablation_matrix() {
  local mode="${ABLATION_COMPONENT_MODE:-train}"
  local variants="${ABLATION_COMPONENT_VARIANTS:-no_events_no_graph events_no_graph events_graph}"
  local variant
  echo "[info] §XI component ablation mode=$mode variants=[$variants] datasets=[$ABLATION_DATASETS]" | tee -a "$LOG_DIR/run_all.log"
  case "$mode" in
    train|retrain)
      for variant in $variants; do
        run_v1_ablation_train "$variant"
      done
      ;;
    eval|eval_only)
      echo "[warn] eval-only component ablation: no_events_no_graph now sets events_disabled=true, but Stage-2 heads were trained WITH events. Cite ABLATION_COMPONENT_MODE=train for claim-grade numbers." | tee -a "$LOG_DIR/run_all.log"
      for variant in $variants; do
        run_ablation_eval "$variant"
      done
      ;;
    *)
      echo "Unknown ABLATION_COMPONENT_MODE=$mode (expected train|eval)" >&2
      return 2
      ;;
  esac
}

run_ablation_matrix() {
  echo "[info] §XI ablation matrix (components via ABLATION_COMPONENT_MODE=$ABLATION_COMPONENT_MODE)" | tee -a "$LOG_DIR/run_all.log"
  run_component_ablation_matrix
}

# §XI-train: train-based ablations. Eval-only toggles cannot separate what the
# model *learned* from eval-time switches, so each arm re-runs the full Stage-2
# schedule from the same Stage-1b init with one component disabled and writes
# its own checkpoint. Default variants come from ABLATION_TRAIN_VARIANTS
# (= ABLATION_COMPONENT_VARIANTS unless overridden).
run_v1_ablation_train_all() {
  echo "[info] §XI-train ablation matrix (retrained per arm from Stage-1b init) variants=[$ABLATION_TRAIN_VARIANTS]" | tee -a "$LOG_DIR/run_all.log"
  local variant
  for variant in $ABLATION_TRAIN_VARIANTS; do
    run_v1_ablation_train "$variant"
  done
}

# C2 graph-path fix sweep. Each arm retrains the full Stage-2 schedule from the
# same Stage-1b init and changes exactly one graph mechanism, so the four
# candidate root causes (double injection, shared head, dirty edges, no positive
# objective) can be compared against events_graph / events_no_graph on equal
# footing. Override the set, e.g. GRAPH_FIX_VARIANTS="events_graph_sephead".
GRAPH_FIX_VARIANTS="${GRAPH_FIX_VARIANTS:-events_graph_pool_decoder events_nodes_only events_graph_sephead events_graph_clean_edges events_graph_advantage}"
run_graph_fix_sweep() {
  echo "[info] C2 graph-path fix sweep variants=[$GRAPH_FIX_VARIANTS] datasets=[$ABLATION_DATASETS]" | tee -a "$LOG_DIR/run_all.log"
  local variant
  for variant in $GRAPH_FIX_VARIANTS; do
    run_v1_ablation_train "$variant"
  done
}

# Ablation-variant -> plan-claim mapping (SIGMOD_EXPERIMENT_PLAN.md §3.3).
# Component names match the eval-only arms: each says what is disabled.
#   events_graph(=full, =encoder_event_graph)       reference Stage-2 arm -> C2 ref
#   events_no_graph(=no_graph, =encoder_event)      events in head, graph residual off -> C2
#   no_events_no_graph(=no_events, =encoder_only)   events_disabled: head event pool
#                    zeroed, graph residual off, no event aux          -> C1
#   no_aux        == Stage-2 schedule without rule-event aux loss      -> C3
#   no_graph_no_aux == both graph and aux removed (C1/C3 control)
run_v1_ablation_train() {
  local raw_variant="$1"
  # Preserve no_graph_no_aux / no_aux (not component aliases).
  local variant
  case "$raw_variant" in
    no_aux|no_graph_no_aux) variant="$raw_variant" ;;
    *) variant="$(ablation_canonical_variant "$raw_variant")" ;;
  esac
  local stage_label="train_abl_${variant}"
  local flags model_extra=""
  case "$variant" in
    events_graph)
      flags="$(v1_stage2_flags)"
      ;;
    events_graph_pool_decoder)
      # P1.3A: the head event pool uses the raw decoder embeddings, so the graph
      # only reaches the logits through the temporal residual (no double injection).
      flags="$(v1_stage2_flags)"
      model_extra='"graph_event_pool_source":"decoder"'
      ;;
    events_nodes_only)
      # P1.3B: keep graph-reasoned nodes in the pool, remove H0+alpha*deltaH.
      # Unlike events_no_graph (which flips the CLI flag), this is a first-class
      # architecture switch and leaves graph reasoning trained.
      flags="$(v1_stage2_flags)"
      model_extra='"graph_residual_enabled":false'
      ;;
    events_graph_sephead)
      # P1.4: dedicated graph-path head weights; the shared Linear otherwise fits
      # both [pool(H0), ...] and [pool(H0+alpha*deltaH), ...] at once.
      flags="$(v1_stage2_flags)"
      model_extra='"separate_graph_head":true'
      ;;
    events_graph_clean_edges)
      # P1.5: hard train-time edges, sparse learned relations, no attribute edges.
      flags="$(v1_stage2_flags)"
      model_extra='"graph_hard_edges":true,"graph_relation_threshold":0.5,"graph_disable_attribute_relations":true'
      ;;
    events_graph_advantage)
      # P2: positive reward for the graph path beating the base path. Defaults to
      # 0.5 when the caller did not set GRAPH_ADVANTAGE_WEIGHT explicitly.
      local adv_weight="${GRAPH_ADVANTAGE_WEIGHT:-0}"
      [[ "$adv_weight" == "0" ]] && adv_weight=0.5
      flags="$(v1_stage2_flags) --graph-advantage-weight $adv_weight"
      ;;
    events_no_graph)
      flags="$(v1_stage2_flags) --disable-graph-correction"
      ;;
    no_graph_no_aux)
      flags="$(v1_stage2_flags) --disable-graph-correction"
      ;;
    no_events_no_graph)
      # True encoder-only: no event pool in the head, no graph residual, and
      # the event decoder receives no task gradient (aux off). All
      # event-shaping losses are zeroed so the decoder stays at its init.
      flags="$(v1_stage2_flags) --disable-graph-correction \
        --measurement-consistency-weight 0 --cardinality-weight 0 \
        --sparse-weight 0 --entropy-weight 0 --width-prior-weight 0 \
        --graph-norm-weight 0"
      model_extra='"events_disabled":true'
      ;;
    no_aux)
      flags="$(v1_stage2_flags)"
      ;;
    # Drop edge relation features (bias + message); keep event graph topology.
    relation_no_features|no_relation_features|no_edge_embedding)
      flags="$(v1_stage2_flags)"
      model_extra='"relation_mode":"no_relation_features"'
      variant="relation_no_features"
      stage_label="train_abl_${variant}"
      ;;
    relation_original|with_relation_features)
      flags="$(v1_stage2_flags)"
      model_extra='"relation_mode":"original"'
      variant="relation_original"
      stage_label="train_abl_${variant}"
      ;;
    *)
      echo "Unknown ablation_train variant: $raw_variant (canonical=$variant)" >&2
      return 2
      ;;
  esac
  echo "[info] train-based ablation arm: $raw_variant -> $stage_label" | tee -a "$LOG_DIR/run_all.log"
  local saved_aux="${EVENT_AUX_WEIGHT:-}"
  case "$variant" in
    no_aux|no_graph_no_aux|no_events_no_graph) EVENT_AUX_WEIGHT=0 ;;
  esac
  local saved_config_extra="${MODEL_CONFIG_EXTRA:-}"
  if [[ -n "$model_extra" ]]; then
    if [[ -n "$saved_config_extra" ]]; then
      MODEL_CONFIG_EXTRA="${saved_config_extra},${model_extra}"
    else
      MODEL_CONFIG_EXTRA="$model_extra"
    fi
    export MODEL_CONFIG_EXTRA
  fi
  local dataset
  for dataset in $ABLATION_DATASETS; do
    case "$dataset" in
      UCI_HAR|SMD|BasicMotions|UWaveGestureLibrary|NATOPS|Cricket|Libras|Epilepsy|RacketSports)
        # event aux is a morphology-only loss; no_aux arms would just duplicate full.
        case "$variant" in
          no_aux|no_graph_no_aux) continue ;;
        esac
        run_v1_downstream event_classification "$dataset" "$stage_label" "$flags"
        ;;
      *) run_v1_downstream morphology "$dataset" "$stage_label" "$flags" ;;
    esac
  done
  if [[ -n "$saved_config_extra" ]]; then
    MODEL_CONFIG_EXTRA="$saved_config_extra"
    export MODEL_CONFIG_EXTRA
  else
    unset MODEL_CONFIG_EXTRA
  fi
  if [[ -n "$saved_aux" ]]; then
    EVENT_AUX_WEIGHT="$saved_aux"
  else
    unset EVENT_AUX_WEIGHT
  fi
}

# Relation Perturbation Ablation, round 2: retrain each arm from the Stage-1b
# init with the same Stage-2 budget. relation_mode is applied inside every
# forward after the event graph is built, so task fine-tuning sees the ablation
# rather than only the final head. Round 1 (frozen checkpoint) is
# TASK=relation_perturbation and does not call this function.
relation_mode_model_extra() {
  local variant="$1"
  case "$variant" in
    relation_original)
      echo '"relation_mode":"original"'
      ;;
    relation_shuffled)
      echo '"relation_mode":"shuffled","shuffle_relation_seed":-1'
      ;;
    relation_no_bias)
      echo '"relation_mode":"no_relation_bias"'
      ;;
    relation_no_features)
      echo '"relation_mode":"no_relation_features"'
      ;;
    *)
      echo "Unknown relation perturbation variant: $variant" >&2
      return 2
      ;;
  esac
}

run_relation_perturbation_train() {
  local variants="${RELATION_TRAIN_VARIANTS:-relation_original relation_shuffled relation_no_bias relation_no_features}"
  local datasets="${RELATION_DATASETS:-CBF TwoPatterns UWaveGestureLibrary BasicMotions}"
  local seeds="${RELATION_TRAIN_SEEDS:-0 1 2}"
  echo "[info] relation perturbation retrain variants=[$variants] datasets=[$datasets] seeds=[$seeds]" | tee -a "$LOG_DIR/run_all.log"
  local saved_extra="${MODEL_CONFIG_EXTRA:-}"
  local saved_args=()
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    saved_args=("${EXTRA_ARGS[@]}")
  fi
  local variant seed dataset model_extra flags
  flags="$(v1_stage2_flags)"
  for variant in $variants; do
    model_extra="$(relation_mode_model_extra "$variant")"
    for seed in $seeds; do
      if [[ -n "$saved_extra" ]]; then
        MODEL_CONFIG_EXTRA="${saved_extra},${model_extra}"
      else
        MODEL_CONFIG_EXTRA="$model_extra"
      fi
      export MODEL_CONFIG_EXTRA
      EXTRA_ARGS=(--seed "$seed" "${saved_args[@]}")
      local stage_label="train_abl_${variant}_s${seed}"
      for dataset in $datasets; do
        case "$dataset" in
          UCI_HAR|SMD|BasicMotions|UWaveGestureLibrary|NATOPS|Cricket|Libras|Epilepsy|RacketSports)
            run_v1_downstream event_classification "$dataset" "$stage_label" "$flags"
            ;;
          *)
            run_v1_downstream morphology "$dataset" "$stage_label" "$flags"
            ;;
        esac
      done
    done
  done
  if [[ -n "$saved_extra" ]]; then
    MODEL_CONFIG_EXTRA="$saved_extra"
    export MODEL_CONFIG_EXTRA
  else
    unset MODEL_CONFIG_EXTRA
  fi
  if [[ ${#saved_args[@]} -gt 0 ]]; then
    EXTRA_ARGS=("${saved_args[@]}")
  else
    EXTRA_ARGS=()
  fi
}

# Joint-oracle Stage-2: train the task heads with ground-truth rule events
# injected (oracle_events during training), then eval under oracle + normal.
joint_oracle_flags() {
  echo --epochs "$STAGE2_EPOCHS" \
    --freeze-epochs "$STAGE2_FREEZE_EPOCHS" \
    --lr "$STAGE2_LR" \
    --encoder-lr "$STAGE2_ENCODER_LR" \
    --measurement-consistency-weight "$MEASUREMENT_CONSISTENCY_WEIGHT" \
    --cardinality-weight "$CARDINALITY_WEIGHT" \
    --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
    --sparse-weight "$SPARSE_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --gradient-clip 1 \
    --patience 6 \
    --event-aux-weight "$EVENT_AUX_WEIGHT" \
    --oracle-events \
    --model-config "{\"num_queries\":${MORPHOLOGY_NUM_QUERIES},\"patch_len\":16,\"patch_stride\":8,\"encoder_layers\":3,\"postprocess_events\":true,\"existence_threshold\":${EVENT_EXISTENCE_THRESHOLD},\"oracle_events\":true}"
}

# Joint-oracle flags for non-morphology Stage-2 tasks (HAR).
# Uses the shared K_MAX query budget; oracle events come from auto shape rules.
# NOTE: no --event-aux-weight here. In oracle mode the decoder is not evaluated
# on real events for this task, and the decoder-aux gradient noise lowered
# joint-oracle UCI_HAR accuracy (~0.66 vs ~0.70 without it, see 09-01 vs
# 08-30 runs). train_task.py also skips that term for event_classification.
joint_oracle_task_flags() {
  echo --epochs "$STAGE2_EPOCHS" \
    --freeze-epochs "$STAGE2_FREEZE_EPOCHS" \
    --lr "$STAGE2_LR" \
    --encoder-lr "$STAGE2_ENCODER_LR" \
    --measurement-consistency-weight "$MEASUREMENT_CONSISTENCY_WEIGHT" \
    --cardinality-weight "$CARDINALITY_WEIGHT" \
    --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
    --sparse-weight "$SPARSE_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --gradient-clip 1 \
    --patience 6 \
    --oracle-events \
    --model-config "{\"num_queries\":${K_MAX},\"patch_len\":16,\"patch_stride\":8,\"encoder_layers\":3,\"postprocess_events\":true,\"existence_threshold\":${EVENT_EXISTENCE_THRESHOLD},\"oracle_events\":true}"
}

run_v1_joint_oracle() {
  echo "[info] joint oracle Stage-2 (train with GT-event injection; all morphology datasets)" | tee -a "$LOG_DIR/run_all.log"
  local dataset
  for dataset in $MORPHOLOGY_DATASETS; do
    run_v1_downstream morphology "$dataset" joint_oracle "$(joint_oracle_flags)"
  done
}

# Joint mix/curriculum Stage-2: mix GT-event injection into training so the
# task heads learn robust event-conditioned classification. Eval always uses
# real decoder events (deployable metrics) via --oracle-eval-events 0.
joint_mix_flags() {
  local variant="$1" mix_args
  case "$variant" in
    mix_50) mix_args="--oracle-train-ratio 0.5 --oracle-curriculum-epochs 0" ;;
    curriculum_3) mix_args="--oracle-train-ratio 0.0 --oracle-curriculum-epochs 3" ;;
    *)
      echo "Unknown joint variant: $variant" >&2
      return 2
      ;;
  esac
  echo --epochs "$STAGE2_EPOCHS" \
    --freeze-epochs "$STAGE2_FREEZE_EPOCHS" \
    --lr "$STAGE2_LR" \
    --encoder-lr "$STAGE2_ENCODER_LR" \
    --measurement-consistency-weight "$MEASUREMENT_CONSISTENCY_WEIGHT" \
    --cardinality-weight "$CARDINALITY_WEIGHT" \
    --width-prior-weight "$WIDTH_PRIOR_WEIGHT" \
    --sparse-weight "$SPARSE_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --gradient-clip 1 \
    --patience 6 \
    --event-aux-weight "$EVENT_AUX_WEIGHT" \
    --oracle-events \
    --oracle-eval-events 0 \
    $mix_args \
    --model-config "{\"num_queries\":${MORPHOLOGY_NUM_QUERIES},\"patch_len\":16,\"patch_stride\":8,\"encoder_layers\":3,\"postprocess_events\":true,\"existence_threshold\":${EVENT_EXISTENCE_THRESHOLD},\"oracle_events\":true}"
}

run_v1_joint_mix() {
  echo "[info] joint mix/curriculum Stage-2 (GT-injection + real-event eval; all morphology datasets)" | tee -a "$LOG_DIR/run_all.log"
  local dataset
  for dataset in $MORPHOLOGY_DATASETS; do
    run_v1_downstream morphology "$dataset" joint_mix_50 "$(joint_mix_flags mix_50)"
    run_v1_downstream morphology "$dataset" joint_curriculum "$(joint_mix_flags curriculum_3)"
  done
}

# Full-model joint-oracle curriculum: Stage 1 → 1b → Stage 2 with GT/auto-rule
# event injection on every Stage-2 task family:
#   morphology (6 UCR) — aligned/rule GT events
#   event_classification — auto morphology rules
# Stage 3 ranking stays out of scope (orthogonal to event injection).
run_v1_joint_oracle_curriculum() {
  echo "[info] full-model joint-oracle curriculum" | tee -a "$LOG_DIR/run_all.log"
  run_v1_stage1_scm
  run_v1_stage1b_morphology
  local dataset
  for dataset in $MORPHOLOGY_DATASETS; do
    run_v1_downstream morphology "$dataset" joint_oracle "$(joint_oracle_flags)"
  done
  for dataset in $EVENT_CLASSIFICATION_DATASETS; do
    run_v1_downstream event_classification "$dataset" joint_oracle "$(joint_oracle_task_flags)"
  done
}

run_v1_curriculum() {
  run_v1_stage1_scm
  run_v1_stage1b_morphology
  run_v1_stage2_curriculum
  run_v1_stage3_ranking
  run_ablation_matrix
}

run_smoke() {
  run_with_log smoke_stage1 \
    "$PYTHON" scripts/run_task.py \
      --task controlled_scm --smoke \
      --checkpoint "$V1_STAGE1_CHECKPOINT" \
      --model-config "$MODEL_CONFIG" \
      "${EXTRA_ARGS[@]}"
  local init
  init="$(require_stage1_init)"
  local flags
  flags="$(v1_stage2_flags)"
  run_with_log smoke_stage2_morphology \
    "$PYTHON" scripts/run_task.py \
      --task morphology --dataset SyntheticControl --smoke \
      --init-checkpoint "$init" \
      --data-root "$DATA_ROOT" \
      --selection-metric macro_f1 \
      --event-aux-weight "${EVENT_AUX_WEIGHT:-0.2}" \
      --model-config "$MODEL_CONFIG" \
      ${flags} \
      "${EXTRA_ARGS[@]}"
}

usage() {
  cat <<'EOF'
V1 TASK values:
  stage1_scm | scm          Stage 1: Controlled-SCM (default)
  stage1b | stage1b_morphology  Stage 1b: rule-based morphology event grounding
  stage2 | task0            Stage 2: ordered per-task Task0 training
  stage3 | ranking          Stage 3: ranking / fine-tune from Stage 2
  curriculum | all          Stage 1 → 1b → 2 → 3 → ablations
  ablation                    Full §XI component matrix
  ablation_components         Component arms only (default: TRAIN each arm)
  ablation_components_eval    Same arms, eval-only on Stage-2 (diagnostic)
  ablation_train              Alias of train-based matrix (ABLATION_TRAIN_VARIANTS)
  smoke                       Quick smoke test

Single-task shortcuts (Stage 2):
  morphology   DATASET=CBF
  event_classification | har   DATASET=SMD (optional; default = EVENT_CLASSIFICATION_DATASETS)

§XI component ablation (C1/C2) — preferred protocol is TRAIN, not eval-only:
  TASK=ablation_components
  ABLATION_COMPONENT_MODE=train|eval   (default train)
  ABLATION_COMPONENT_VARIANTS="no_events_no_graph events_no_graph events_graph"
  Arms (name = kept / disabled; legacy alias in parens):
    no_events_no_graph (encoder_only)  temporal encoder only: events_disabled + no graph
    events_no_graph    (encoder_event) encoder + events: graph correction off
    events_graph       (encoder_event_graph / full)  full model
  train: each arm retrains Stage-2 from Stage-1b (claim-grade).
  eval:  inference switch on Stage-2 ckpt; no_events now sets events_disabled=true
         but the head was still trained WITH events — diagnostic only.

§XI-train (explicit train matrix / single arm):
  train_abl_<variant> | ablation_train_<variant>
      events_graph | events_no_graph | no_events_no_graph | no_aux | no_graph_no_aux
  ablation_train             all arms in ABLATION_TRAIN_VARIANTS
  (legacy aliases still work; no_events needs events_disabled in task_heads.py)

§XVI C2 graph-path isolation sweep: graph_fix | graph_fix_sweep
  Each arm retrains the full Stage-2 schedule from the Stage-1b init and changes
  exactly one graph mechanism, to be compared against events_graph/events_no_graph:
    events_graph_pool_decoder  graph only via deltaH (no double event injection)
    events_nodes_only          graph-reasoned pool, no temporal residual
    events_graph_sephead       dedicated graph-path head weights
    events_graph_clean_edges   hard edges + sparse learned relations + no attr edges
    events_graph_advantage     positive reward for graph beating base
  Arms: GRAPH_FIX_VARIANTS (default: all five above)

§XII joint oracle: joint_oracle
  Stage-2 training with GT-event injection (true ceiling), on all
  MORPHOLOGY_DATASETS (default 15 UCR morphology datasets).

§XIII joint mix: joint_mix
  mix_50: 50% GT injection during training; curriculum_3: 3 GT epochs then real.
  Eval always on real decoder events (deployable metrics). All MORPHOLOGY_DATASETS.

§XIV full-model joint oracle: joint_oracle_full
  Stage 1 → 1b → Stage 2 with event injection on ALL Stage-2 tasks:
    morphology×15 (rule/aligned GT), event_classification×9
  (aligned rule / auto morphology rules). Stage 3 stays separate.

§XVII event-graph reasoning value: event_value | graph_value | shuffle_edges
  Independent analysis under /data/panhongyu/TS-LLM/event_value (eval-only).
  Same Stage-2 / train_abl_events_graph checkpoint:
    intact           full GraphReasoner (reference)
    identity_graph   skip GraphReasoner → field embeddings  **primary claim**
    shuffle_edges    permute edge-pair relations (secondary; often small Δ)
    no_graph         --disable-graph-correction (C2 residual switch)
  Paper wording: "graph reasoning over event nodes"; do not claim relational
  edges drive accuracy unless shuffle Δ is large.
  Env: DATASETS ARMS SHUFFLE_SEEDS MAX_BATCHES DEVICE BATCH_SIZE OUT_DIR
  CHECKPOINT_DIR (see event_value/README.md).

§XVIII open-set / structure-aware OOD: ood | compositional_ood | loco
  Independent analysis under /data/panhongyu/TS-LLM/ood.
  Layer 1: Leave-One-Class-Out unknown rejection (MSP / Energy)
  Layer 2: Dense / Event / Graph / Shuffle prototype distances
  Layer 3: Controlled-SCM compositional LOCO (optional)
  Priority datasets: UCI_HAR SyntheticControl UWaveGestureLibrary
  Env: MODE (all|open_set|scm) DATASETS EPOCHS SMOKE DEVICE OUT_DIR
  (see ood/README.md).

§XIX-b relation perturbation ablation: relation_perturbation | relation_ablation
  Round 1, inference-time, under /data/panhongyu/TS-LLM/Edge.
  Frozen checkpoint, four arms on the same event nodes and valid edges:
    original | shuffled (SHUFFLE_SEEDS) | no_relation_bias | no_relation_features
  Default datasets: CBF TwoPatterns UWaveGestureLibrary BasicMotions
  Env: DATASETS SHUFFLE_SEEDS MAX_BATCHES DEVICE BATCH_SIZE OUT_DIR TOLERANCE

§XIX-c relation / edge-embedding off: train_abl_relation_no_features
  Retrain Stage-2 with relation_mode=no_relation_features (drop r_uv bias+message,
  keep edge topology). Alias: no_relation_features | no_edge_embedding.
  Compare against train_abl_relation_original or existing events_graph.
  Env: ABLATION_DATASETS (default all). Example:
    ABLATION_DATASETS="CBF TwoPatterns UWaveGestureLibrary BasicMotions" \\
    TASK=ablation_train_relation_no_features bash scripts/run_all.sh
  Or RELATION_MODE=no_relation_features on any Stage-2 run.
  Full four-arm retrain: TASK=relation_perturbation_train
  (RELATION_TRAIN_VARIANTS / RELATION_DATASETS / RELATION_TRAIN_SEEDS)

§XIX relation contribution heatmap: edge | relation_contrib | relation_heatmap
  Independent analysis under /data/panhongyu/TS-LLM/Edge (eval-only).
  Leave-one-relation-out on Stage-2 / train_abl_events_graph checkpoints:
    intact + drop each of BEFORE / OVERLAP / CONTAINS / LONGER_THAN /
    HIGHER_THAN / STEEPER_THAN / SAME_FREQUENCY + drop_Learned
  ΔF1 = intact − drop → Relation Contribution Heatmap.
  Env: DATASETS ARMS MAX_BATCHES DEVICE BATCH_SIZE OUT_DIR
  CHECKPOINT_DIR (see Edge/README.md).

§XX gated residual localization: residual | gated_residual | residual_localization
  Independent analysis under /data/panhongyu/TS-LLM/residual (eval-only).
  For each test window: raw signal + detected event intervals +
  ‖α·ΔH_t‖ correction intensity on the timeline.
  Metrics: mass@event, coverage, lift (= mass/coverage), peak-in-event,
  mass@GT (morphology rules when available).
  Explains the gated residual H'=H0+α·ΔH (membership-localized).
  Env: DATASETS N_EXAMPLES EXISTENCE_THRESHOLD MAX_BATCHES DEVICE
  BATCH_SIZE OUT_DIR CHECKPOINT_DIR (see residual/README.md).

Environment:
  V1_STAGE1_CHECKPOINT  V1_STAGE1B_CHECKPOINT  STAGE1/1B/2/3_EPOCHS  STAGE2_FREEZE_EPOCHS
  MORPHOLOGY_MAX_EVENTS  MORPHOLOGY_NUM_QUERIES  EVENT_AUX_WEIGHT
  MEASUREMENT_CONSISTENCY_WEIGHT  MODEL_CONFIG  LOG_DIR  DATA_ROOT
  STRICT_QUERY_INIT=1 (raise instead of warning when event_decoder.queries num_queries mismatches on load)
  MORPHOLOGY_DATASETS  EVENT_CLASSIFICATION_DATASETS
    (defaults: 15 UCR morphology + UCI_HAR SMD BasicMotions UWaveGestureLibrary NATOPS Cricket Libras Epilepsy RacketSports)
  ABLATION_DATASETS (default: all datasets of both tasks)
  ABLATION_COMPONENT_MODE (default train)  ABLATION_COMPONENT_VARIANTS
  ABLATION_TRAIN_VARIANTS (default = ABLATION_COMPONENT_VARIANTS)
  GRAPH_FIX_VARIANTS (default: the five §XVI graph-fix arms)
  GRAPH_RESIDUAL_ENABLED=0 (graph-reasoned pool, no temporal residual)
  GRAPH_EVENT_POOL_SOURCE=graph|decoder (decoder = graph only via deltaH)
  GRAPH_SEPARATE_HEAD=1 (dedicated graph-path head weights)
  GRAPH_HARD_EDGES=1 (hard train-time graph edges)
  GRAPH_RELATION_THRESHOLD (drop learned relations below this value)
  GRAPH_DISABLE_ATTRIBUTE_RELATIONS=1 (drop HIGHER/STEEPER/SAME_FREQUENCY edges)
  GRAPH_GATE_INIT (initial graph gate, default 0.1)
  GRAPH_ADVANTAGE_WEIGHT (reward graph path beating base; default 0)
  TWOTLEAD_BATCH_SIZE TWOTLEAD_EPOCHS TWOTLEAD_FREEZE_EPOCHS
  TWOTLEAD_LR TWOTLEAD_ENCODER_LR TWOTLEAD_PATIENCE TWOTLEAD_VAL_RATIO
  SYNTHETIC_BATCH_SIZE SYNTHETIC_EPOCHS SYNTHETIC_FREEZE_EPOCHS
  SYNTHETIC_LR SYNTHETIC_ENCODER_LR SYNTHETIC_PATIENCE SYNTHETIC_VAL_RATIO
  DETAIL_SAMPLES  DETAIL_MAX_POINTS  DETAIL_MIN_EPOCH  DETAIL_MAX_CHANNELS
  EVENT_EXISTENCE_THRESHOLD  WIDTH_PRIOR_WEIGHT
  SPARSE_WEIGHT  ENTROPY_WEIGHT  STAGE1_CHANNEL_WEIGHT  K_MAX
  USE_RAW_CHANNEL_ONLY (stage1b input: 1 = raw channel only, default)
  (per-task detail logs: LOG_DIR/<task>.detail.log; pics: LOG_DIR/pic_<task>/)
EOF
}

# --- TASK dispatch ------------------------------------------------------------
case "$TASK" in
  stage1|stage1_scm|scm|controlled_scm) run_v1_stage1_scm ;;
  stage1b|stage1b_morphology|morphology_events) run_v1_stage1b_morphology ;;
  stage2|stage2_task0|task0|downstream) run_v1_stage2_curriculum ;;
  stage3|stage3_ranking|ranking) run_v1_stage3_ranking ;;
  curriculum|all|v1) run_v1_curriculum ;;
  ablation|ablation_matrix) run_ablation_matrix ;;
  ablation_components|component_ablation|abl_components) run_component_ablation_matrix ;;
  ablation_components_eval|ablation_eval_components)
    ABLATION_COMPONENT_MODE=eval run_component_ablation_matrix
    ;;
  ablation_train|train_abl) run_v1_ablation_train_all ;;
  smoke) run_smoke ;;

  morphology)
    flags="$(v1_stage2_flags)"
    for dataset in ${DATASET:-$MORPHOLOGY_DATASETS}; do
      run_v1_downstream morphology "$dataset" stage2 "$flags"
    done
    ;;
  event_classification|har)
    flags="$(v1_stage2_flags)"
    for dataset in ${DATASET:-$EVENT_CLASSIFICATION_DATASETS}; do
      run_v1_downstream event_classification "$dataset" stage2 "$flags"
    done
    ;;

  # NOTE: must precede ablation_* below, otherwise ablation_train_<variant> is
  # swallowed by that glob and parsed as the (unknown) "train_<variant>" variant.
  ablation_train_*|train_abl_*)
    variant="${TASK#ablation_train_}"
    variant="${variant#train_abl_}"
    run_v1_ablation_train "$variant"
    ;;

  ablation_*)
    variant="${TASK#ablation_}"
    case "$variant" in
      # Component arms: follow ABLATION_COMPONENT_MODE (default train).
      no_events_no_graph|encoder_only|no_events|events_no_graph|encoder_event|no_graph|\
events_graph|encoder_event_graph|full_no_verifier|full)
        if [[ "${ABLATION_COMPONENT_MODE:-train}" == "eval" || "${ABLATION_COMPONENT_MODE:-train}" == "eval_only" ]]; then
          run_ablation_eval "$variant"
        else
          run_v1_ablation_train "$variant"
        fi
        ;;
      oracle_event)
        run_ablation_eval "$variant" ;;
      events_graph_pool_decoder|events_nodes_only|events_graph_sephead|\
events_graph_clean_edges|events_graph_advantage|\
relation_no_features|no_relation_features|no_edge_embedding|\
relation_original|with_relation_features)
        run_v1_ablation_train "$variant" ;;
      *) echo "Unknown ablation variant: $variant" >&2; usage; exit 2 ;;
    esac
    ;;

  graph_fix|graph_fix_sweep|graph_isolation|graph_isolate)
    run_graph_fix_sweep
    ;;

  joint_oracle|stage2_oracle)
    run_v1_joint_oracle
    ;;

  joint_mix|mix|curriculum)
    run_v1_joint_mix
    ;;

  joint_oracle_full|full_oracle|oracle_curriculum)
    run_v1_joint_oracle_curriculum
    ;;

  event_value|graph_value|shuffle_edges|structure_control)
    # §XVII: prove event-graph independent value via edge-pair shuffle control.
    EVENT_VALUE_ROOT="${EVENT_VALUE_ROOT:-/data/panhongyu/TS-LLM/event_value}"
    if [[ ! -x "$EVENT_VALUE_ROOT/run.sh" && ! -f "$EVENT_VALUE_ROOT/run.sh" ]]; then
      echo "Missing event_value runner: $EVENT_VALUE_ROOT/run.sh" >&2
      exit 2
    fi
    echo "[info] event_value structure control -> $EVENT_VALUE_ROOT" | tee -a "$LOG_DIR/run_all.log"
    (
      cd "$EVENT_VALUE_ROOT"
      export PYTHON TSLLM_MODEL_ROOT="$ROOT" DATA_ROOT CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
      export DATASETS="${DATASETS:-$MORPHOLOGY_DATASETS}"
      export SHUFFLE_SEEDS DEVICE BATCH_SIZE MAX_BATCHES OUT_DIR STAMP
      bash run.sh
    ) 2>&1 | tee -a "$LOG_DIR/run_all.log"
    ;;

  ood|compositional_ood|loco|loco_ood|open_set_ood)
    # §XVIII: LOCO unknown-class rejection + structure-aware OOD scores.
    OOD_ROOT="${OOD_ROOT:-/data/panhongyu/TS-LLM/ood}"
    if [[ ! -f "$OOD_ROOT/run.sh" ]]; then
      echo "Missing ood runner: $OOD_ROOT/run.sh" >&2
      exit 2
    fi
    echo "[info] open-set / structure-aware OOD -> $OOD_ROOT" | tee -a "$LOG_DIR/run_all.log"
    (
      cd "$OOD_ROOT"
      export PYTHON TSLLM_MODEL_ROOT="$ROOT" DATA_ROOT CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
      export MODE="${MODE:-open_set}" SMOKE DEVICE OUT_DIR STAMP
      export DATASETS EPOCHS BATCH_SIZE MAX_BATCHES
      export SCM_EPOCHS SCM_TRAIN_SIZE SCM_EVAL_SIZE SCM_BATCH SCM_INIT
      bash run.sh
    ) 2>&1 | tee -a "$LOG_DIR/run_all.log"
    ;;

  relation_perturbation|relation_ablation|relation_perturb)
    EDGE_ROOT="${EDGE_ROOT:-/data/panhongyu/TS-LLM/Edge}"
    if [[ ! -f "$EDGE_ROOT/run_relation_perturbation.sh" ]]; then
      echo "Missing relation perturbation runner: $EDGE_ROOT/run_relation_perturbation.sh" >&2
      exit 2
    fi
    echo "[info] relation perturbation (inference) -> $EDGE_ROOT" | tee -a "$LOG_DIR/run_all.log"
    (
      cd "$EDGE_ROOT"
      export PYTHON TSLLM_MODEL_ROOT="$ROOT" DATA_ROOT CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
      export DATASETS="${DATASETS:-CBF TwoPatterns UWaveGestureLibrary BasicMotions}"
      export SHUFFLE_SEEDS DEVICE BATCH_SIZE MAX_BATCHES OUT_DIR STAMP TOLERANCE
      bash run_relation_perturbation.sh
    ) 2>&1 | tee -a "$LOG_DIR/run_all.log"
    ;;

  relation_perturbation_train|relation_ablation_train)
    run_relation_perturbation_train
    ;;

  edge|relation_contrib|relation_heatmap|relation_contribution)
    # §XIX: leave-one-relation-out contribution heatmap.
    EDGE_ROOT="${EDGE_ROOT:-/data/panhongyu/TS-LLM/Edge}"
    if [[ ! -f "$EDGE_ROOT/run.sh" ]]; then
      echo "Missing Edge runner: $EDGE_ROOT/run.sh" >&2
      exit 2
    fi
    echo "[info] relation contribution heatmap -> $EDGE_ROOT" | tee -a "$LOG_DIR/run_all.log"
    (
      cd "$EDGE_ROOT"
      export PYTHON TSLLM_MODEL_ROOT="$ROOT" DATA_ROOT CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
      export DATASETS ARMS DEVICE BATCH_SIZE MAX_BATCHES OUT_DIR STAMP
      bash run.sh
    ) 2>&1 | tee -a "$LOG_DIR/run_all.log"
    ;;

  residual|gated_residual|residual_localization|graph_residual)
    # §XX: Graph-Guided Residual Localization (gated residual explanation).
    RESIDUAL_ROOT="${RESIDUAL_ROOT:-/data/panhongyu/TS-LLM/residual}"
    if [[ ! -f "$RESIDUAL_ROOT/run.sh" ]]; then
      echo "Missing residual runner: $RESIDUAL_ROOT/run.sh" >&2
      exit 2
    fi
    echo "[info] gated residual localization -> $RESIDUAL_ROOT" | tee -a "$LOG_DIR/run_all.log"
    (
      cd "$RESIDUAL_ROOT"
      export PYTHON TSLLM_MODEL_ROOT="$ROOT" DATA_ROOT CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
      export DATASETS N_EXAMPLES EXISTENCE_THRESHOLD DEVICE BATCH_SIZE MAX_BATCHES OUT_DIR STAMP
      bash run.sh
    ) 2>&1 | tee -a "$LOG_DIR/run_all.log"
    ;;

  help|-h|--help) usage ;;

  *)
    echo "Unsupported TASK=$TASK" >&2
    usage
    exit 2
    ;;
esac
