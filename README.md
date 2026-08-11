# Space Recognition

`space-recognition` restores word boundaries in already-normalised transcriptions. Given `ðəkwɪkbraʊnfoks`, it predicts boundary positions and reconstructs text such as `ðə kwɪk braʊn foks`.

## Pipeline position and storage

Run the projects in this order:

```text
data-normalisation → speech-recognition → space-recognition
```

The gold files used for training must retain their correct spaces. Preparation removes whitespace to construct the paired model input; it does not repeat phonetic normalization. Space-free inference files may come from ASR or another normalisation run.

Lightweight manifests and audit logs stay here:

```text
space-recognition/logs/
├── prep/<language>/<timestamp>/
└── inference/<language>/<timestamp>/
```

Model weights and restored files stay with the language data:

```text
<language-root>/
├── processed/spaces/bilstm-boundary_YYYYMMDD_HHMMSS/
└── space-recognised/YYYY-MM-DD-HH-MM-SS/
```

Source transcription files are never overwritten.

## Requirements

This project must be run in the shared `tcas_asr_python3.10` Conda environment. Create it once with Python 3.10, activate it, and install this project's dependencies:

```bash
conda create --name tcas_asr_python3.10 python=3.10
conda activate tcas_asr_python3.10
python -m pip install -r requirements.txt
```

## 1. Prepare gold pairs

Edit `config/prep/prepare.yaml`. `data_roots` accepts multiple roots and
`prefer` selects every listed format rather than expressing a priority.

```bash
python scripts/prepare_space_training.py --config config/prep/prepare.yaml --dry-run
python scripts/prepare_space_training.py --config config/prep/prepare.yaml
```

Supported sources are EAF, Praat long-text TextGrid, CSV, TSV, and delimited TXT. EAF/TextGrid selectors name tiers. Table selectors name columns or use 1-based column numbers; headerless inputs require numbers. `all` selects every available tier or column.

Preparation collapses whitespace to one ASCII space and removes all whitespace for `input_text`. Identical inputs are kept in one split. If an identical space-free input has different gold targets, every occurrence is excluded and recorded in `conflicts.csv`, because the distinction is impossible to learn from the available input.

The run writes `manifest.csv`, `train.csv`, `dev.csv`, `test.csv`, `conflicts.csv`, and `summary.json`.

## 2. Train

Replace `<prep-run>` in `./logs/prep/yonghe_qiang_01/<prep-run>/...` from the `config/training/train.yaml`, then run:

```bash
python -m src.training.train_boundary --config config/training/train.yaml
```

The default two-layer bidirectional LSTM uses grapheme embeddings and a weighted binary boundary loss. It chooses the checkpoint and insertion threshold by development boundary F1 with early stopping. Each external model run contains:

- `model.pt`, `model_config.json`, and `vocab.json`
- `threshold.json` and the resolved `run_config.json`
- `train_log.tsv` and `test_metrics.json`

Metrics include boundary precision/recall/F1, exact sentence accuracy, and word error rate.

## 3. Restore files

Edit `config/inference/restore.yaml` to select the space-free sources and a trained model run. Validate first, then write timestamped copies:

```bash
python -m src.inference.restore_spaces --config config/inference/restore.yaml --dry-run
python -m src.inference.restore_spaces --config config/inference/restore.yaml
```

Only selected annotation values are replaced. EAF tier and annotation IDs, TextGrid intervals and points, table cells outside selected columns, and all unselected metadata remain in their original positions. Existing whitespace in selected values is removed before prediction, making the command suitable for both fully space-free and partially spaced ASR output.

Inference logs `predictions.csv` with the source path, tier/column locator, original text, model input, and restored text, plus `summary.json`.

TP = TRUE POSITIVE, FP = FALSE POSITIVE, FN = FALSE NEGATIVE, TN = TRUE NEGATIVE

Precision asks "how many spaces were correct"; TP/(TP+FP)
Recall asks "how many spaces did I miss"; TP/(TP+FN)
Boundary F1 asks "given a tolerance of θ, how close is the predicted text to the ground truth"
WER = (S+D+I)/N 
Exact Sense asks "how many sentences are a pure match"
