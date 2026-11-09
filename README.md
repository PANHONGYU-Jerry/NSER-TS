# NSER-TS

NSER-TS reads a multivariate window, decodes a small set of typed events, builds an event graph over those events, and predicts the task label from the encoder state plus a gated residual from the graph. Classification metrics use the task-head prediction (`accuracy`, `macro_f1`) on the test split after the checkpoint is chosen on validation.

Run every command from the repository root.

## Setup

Python 3.10 or newer. A CUDA GPU is assumed for the full schedule; add `--device cpu` to any `scripts/run_task.py` call for a CPU run.

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib aeon scikit-learn
```

`aeon` and `scikit-learn` are used by the multivariate UEA loader. `matplotlib` is used only by the figure scripts.

The shell runner picks up `PYTHON`. Point it at the interpreter you just installed:

```bash
export PYTHON=python
export DATA_ROOT=/path/to/datasets
export CHECKPOINT_DIR="$PWD/checkpoints"
export K_MAX=6
mkdir -p "$DATA_ROOT" "$CHECKPOINT_DIR"
```

`K_MAX=6` is the event-slot budget used for the reported runs. Stage 1, Stage 1b, and the downstream tasks need the same value, otherwise a later stage drops decoder slots when it loads the previous checkpoint.

## Data

Stage 1 generates its own series and does not read `DATA_ROOT`. Everything after that does.

```
$DATA_ROOT/
  CBF/CBF_TRAIN.txt
  CBF/CBF_TEST.txt
  ECG5000/ ...
  UCI_HAR/train/Inertial Signals/
  UCI_HAR/train/y_train.txt
  UCI_HAR/train/subject_train.txt
  UCI_HAR/test/...
  OmniAnomaly-master/ServerMachineDataset/train/machine-1-1.txt
  OmniAnomaly-master/ServerMachineDataset/test/machine-1-1.txt
  OmniAnomaly-master/ServerMachineDataset/test_label/machine-1-1.txt
  BasicMotions/BasicMotions_TRAIN.ts
  BasicMotions/BasicMotions_TEST.ts
  ETT-small/ETTm1.csv          # only for --task forecast
```

UCR files are the official text archives: column 0 is the class label, the remaining columns are the series. The same layout is used for every name in the morphology list below. UCI HAR is the official inertial-signal release. SMD is the OmniAnomaly `ServerMachineDataset` tree; if `train/` and `test/` already sit directly under `DATA_ROOT`, that directory is used as-is. UEA sets are the `.ts` files published with the archive.

Validation is cut from the official training split. The test split is held out and scored once, after the selected checkpoint is reloaded. SyntheticControl uses a 10% validation cut and TwoLeadECG a 30% cut; other classification sets use 20%. Those two sets also use a longer Stage-2 schedule inside `scripts/run_all.sh` (30 and 40 epochs). Leave that override in place when reproducing the tables.

### Rule-event files for Stage 1b

Stage 1b supervises the event decoder with rule-event records, not with human interval labels. Point `DEFAULT_MORPHOLOGY_ROOTS` in `src/data/morphology_events.py` at a directory that contains, for each morphology dataset:

```
<dataset>/train/multi_supervision.jsonl
<dataset>/train/X_features.npy
<dataset>/test/multi_supervision.jsonl
<dataset>/test/X_features.npy
```

`X_features.npy` must be index-aligned with the official `TRAIN`/`TEST` file. Channel 0 is the raw series used for that check. If the files are missing or the alignment check fails, Stage 2 still trains, and the log prints `event_aux` with `"enabled": false`. Stage 1b itself stops without these files.

## Check the install

The unit test builds a tiny model on synthetic SCM batches. It does not download data.

```bash
python -m unittest tests.test_v1
```

A one-epoch Stage-1 pass:

```bash
bash scripts/run_all.sh --smoke
```

Stage 1 plus one SyntheticControl batch. This step reads `$DATA_ROOT/SyntheticControl/`:

```bash
TASK=smoke bash scripts/run_all.sh
```

## Training order

Later stages load the checkpoint written by the previous stage. The paper runs start Stage 2 from the Stage-1b checkpoint.

| Stage | What it trains | Checkpoint |
| --- | --- | --- |
| 1 | event slots on synthetic SCM series | `$CHECKPOINT_DIR/v1_stage1_scm.pt` |
| 1b | the same slots on real morphology series | `$CHECKPOINT_DIR/v1_stage1b_morphology.pt` |
| 2 | one task head per dataset | see below |
| 3 | ranking / fine-tune, trunk frozen except the scorer and the graph gate | `stage3_*.pt` |

Stage 2 file names:

- morphology: `$CHECKPOINT_DIR/stage2_morphology_<Dataset>.pt`
- UCI HAR: `$CHECKPOINT_DIR/stage2_event_classification.pt`
- other event datasets: `$CHECKPOINT_DIR/stage2_event_classification_<Dataset>.pt`

### Paper datasets

Morphology (UCR):

```text
CBF ECG5000 FordA FordB SyntheticControl TwoLeadECG TwoPatterns Wafer
```

`FordB` is supported by the same loader. The main morphology table uses CBF, ECG5000, FordA, SyntheticControl, TwoLeadECG, TwoPatterns, and Wafer.

Event classification:

```text
UCI_HAR SMD BasicMotions UWaveGestureLibrary
```

Trace is a UCR morphology set in this runner (`TASK=morphology DATASET=Trace`), even though some tables group it with the event tasks. Additional UEA names accepted by `TASK=event_classification` are NATOPS, Cricket, Libras, Epilepsy, and RacketSports.

### Commands

```bash
# Stage 1, then Stage 1b
TASK=stage1_scm bash scripts/run_all.sh
TASK=stage1b_morphology bash scripts/run_all.sh

# one morphology dataset
DATASET=CBF TASK=morphology bash scripts/run_all.sh

# one event dataset
DATASET=UCI_HAR TASK=event_classification bash scripts/run_all.sh

# every dataset in the two lists above
MORPHOLOGY_DATASETS="CBF ECG5000 FordA SyntheticControl TwoLeadECG TwoPatterns Wafer" \
EVENT_CLASSIFICATION_DATASETS="UCI_HAR SMD BasicMotions UWaveGestureLibrary" \
TASK=stage2 bash scripts/run_all.sh

# optional: Stage-3 ranking from the Stage-2 checkpoints
TASK=stage3 bash scripts/run_all.sh
```

`TASK=stage2` walks both lists. Set `MORPHOLOGY_DATASETS` and `EVENT_CLASSIFICATION_DATASETS` before a full run, or it will also train the extra archives compiled into the script.

Stage 2 freezes the encoder for 5 epochs, then trains it at `1e-5` while the head stays at `1e-3`, for 20 epochs (patience 6), except the two small UCR sets noted above. Logs go to `log/<timestamp>/`. Each task writes `<name>.log` and, when samples are dumped, `<name>.detail.log`. Figures from a detail log land in `log/<timestamp>/pic_<name>/`.

Extra arguments are forwarded to `scripts/run_task.py`:

```bash
DATASET=CBF TASK=morphology bash scripts/run_all.sh --seed 0 --device cuda
```

The same Stage-2 job without the shell runner:

```bash
python scripts/run_task.py \
  --task morphology \
  --dataset CBF \
  --data-root "$DATA_ROOT" \
  --init-checkpoint "$CHECKPOINT_DIR/v1_stage1b_morphology.pt" \
  --checkpoint "$CHECKPOINT_DIR/stage2_morphology_CBF.pt" \
  --epochs 20 --freeze-epochs 5 \
  --lr 0.001 --encoder-lr 0.00001 \
  --batch-size 32 --patience 6 --gradient-clip 1 \
  --measurement-consistency-weight 0.1 \
  --cardinality-weight 0.05 --sparse-weight 0.05 \
  --entropy-weight 0.01 --width-prior-weight 0.05 \
  --event-aux-weight 0.2 \
  --selection-metric macro_f1 \
  --model-config '{"num_queries":6,"patch_len":16,"patch_stride":8,"encoder_layers":3,"postprocess_events":true,"existence_threshold":0.5}'
```

Forecasting and anomaly detection use the same entry point and the same Stage-1 checkpoint. They are outside the classification curriculum above.

```bash
python scripts/run_task.py --task forecast --dataset ETTm1 \
  --data-root "$DATA_ROOT" \
  --init-checkpoint "$CHECKPOINT_DIR/v1_stage1_scm.pt" \
  --checkpoint "$CHECKPOINT_DIR/forecast_ETTm1.pt"

python scripts/run_task.py --task anomaly \
  --data-root "$DATA_ROOT" \
  --init-checkpoint "$CHECKPOINT_DIR/v1_stage1_scm.pt" \
  --checkpoint "$CHECKPOINT_DIR/anomaly.pt"
```

Forecast reads `$DATA_ROOT/ETT-small/ETTm1.csv` (or `Weather/Weather.csv` with `--dataset Weather`). Anomaly reads TSB-AD: series under `$DATA_ROOT/TSB-AD-U/` or `TSB-AD-M/`, and split lists under `$DATA_ROOT/TSB-AD/`.

## Reading the results

`scripts/run_task.py` prints one JSON object per epoch and a final object on stdout. The shell log stores both. The final object has `"ok": true`, `"checkpoint"`, `"best_epoch"`, `"val_metrics"`, and `"test_metrics"`.

A checkpoint is a dict with `model` (the `state_dict`), `best_epoch`, `best_metrics`, and `selection_metric`. Classification selection uses validation `macro_f1`.

## Ablations

Component ablations retrain Stage 2 from the Stage-1b checkpoint. `ABLATION_DATASETS` chooses the datasets. `ABLATION_COMPONENT_MODE=train` (the default) retrains each arm. `ABLATION_COMPONENT_MODE=eval` applies the same switch on an existing `stage2_*.pt` checkpoint and does not update weights.

The switches are a CLI flag, a field in `--model-config`, or both. The shell `TASK` names below set those fields for you.

| Arm | Switch | What changes |
| --- | --- | --- |
| `no_events_no_graph` | `--disable-graph-correction` and `"events_disabled": true` | Event half of the head input is zero. Graph gate α is 0. Event losses and `--event-aux-weight` are 0. |
| `events_no_graph` | `--disable-graph-correction` | Event pool stays in the head. α is 0, so the label is the base head. |
| `events_nodes_only` | `"graph_residual_enabled": false` | Graph-reasoned events stay in the pool. The residual H₀ + αΔH is removed. |
| `events_graph_pool_decoder` | `"graph_event_pool_source": "decoder"` | The pool uses decoder states. The graph reaches the label only through ΔH. |
| `no_aux` | `--event-aux-weight 0` | Rule-event auxiliary loss is off. The graph path stays on. Morphology only. |
| `events_graph` | none | Reported model: events, graph pool, and residual. |

```bash
ABLATION_DATASETS="CBF UCI_HAR" TASK=ablation_train_no_events_no_graph bash scripts/run_all.sh
ABLATION_DATASETS="CBF UCI_HAR" TASK=ablation_train_events_no_graph bash scripts/run_all.sh
ABLATION_DATASETS="CBF UCI_HAR" TASK=ablation_train_events_nodes_only bash scripts/run_all.sh
ABLATION_DATASETS="CBF UCI_HAR" TASK=ablation_train_events_graph bash scripts/run_all.sh
```

Checkpoints are `train_abl_<arm>_...pt` under `$CHECKPOINT_DIR`. The same switches on one dataset, without the shell names:

```bash
# events off, graph residual off
python scripts/run_task.py --task morphology --dataset CBF \
  --data-root "$DATA_ROOT" \
  --init-checkpoint "$CHECKPOINT_DIR/v1_stage1b_morphology.pt" \
  --checkpoint "$CHECKPOINT_DIR/train_abl_no_events_no_graph_morphology_CBF.pt" \
  --disable-graph-correction --event-aux-weight 0 \
  --measurement-consistency-weight 0 --cardinality-weight 0 \
  --sparse-weight 0 --entropy-weight 0 --width-prior-weight 0 \
  --model-config '{"num_queries":6,"events_disabled":true}'

# events on, graph residual off
python scripts/run_task.py --task morphology --dataset CBF \
  --data-root "$DATA_ROOT" \
  --init-checkpoint "$CHECKPOINT_DIR/v1_stage1b_morphology.pt" \
  --checkpoint "$CHECKPOINT_DIR/train_abl_events_no_graph_morphology_CBF.pt" \
  --disable-graph-correction \
  --model-config '{"num_queries":6}'
```

Any Stage-2 shell run accepts the same knobs as environment variables: `GRAPH_RESIDUAL_ENABLED=0`, `GRAPH_EVENT_POOL_SOURCE=decoder`, `EVENT_AUX_WEIGHT=0`.

`--oracle-events` injects rule events in place of the decoder for ceiling runs:

```bash
ABLATION_DATASETS="CBF" TASK=ablation_oracle_event bash scripts/run_all.sh
```

## Layout

```
scripts/run_task.py            single training or eval job, JSON on stdout
scripts/run_all.sh             stage curriculum, dataset loops, ablations
src/models/                    encoder, event decoder, graph, task heads
src/training/                  Stage 1 / 1b / 2 loops and losses
src/data/                      SCM, UCR, UCI HAR, SMD, UEA, TSB-AD, ETT
src/evaluation/                classification and event metrics
tests/test_v1.py               shape and integration smoke tests
```

The model class is `src.models.GroundedEventGraphReasoner`. `forward(x, time_mask, channel_mask, task=...)` returns `events` and `outputs`.
