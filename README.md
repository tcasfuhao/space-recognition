# Space Recognition

`space-recognition` restores word boundaries in already-normalised transcriptions.
Given `ðəkwɪkbraʊnfoks`, it predicts boundary positions and reconstructs text such
as `ðə kwɪk braʊn foks`.

## Pipeline position and storage

Run the projects in this order:

```text
data-normalisation → speech-recognition → space-recognition
```

The gold files used for training must retain their correct spaces. Preparation
removes whitespace to construct the paired model input; it does not repeat
phonetic normalization. Space-free inference files may come from ASR or another
normalisation run.

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

The existing ASR environment already supplies most dependencies. For a separate
environment:

```bash
python -m pip install -r requirements.txt
```

## 1. Prepare gold pairs

Edit `config/prep/prepare_yq.yaml`. `data_roots` accepts multiple roots and
`prefer` selects every listed format rather than expressing a priority.

```bash
python scripts/prepare_space_training.py --config config/prep/prepare_yq.yaml --dry-run
python scripts/prepare_space_training.py --config config/prep/prepare_yq.yaml
```

Supported sources are EAF, Praat long-text TextGrid, CSV, TSV, and delimited TXT. EAF/TextGrid selectors name tiers. Table selectors name columns or use 1-based column numbers; headerless inputs require numbers. `all` selects every available tier or column.

Preparation collapses whitespace to one ASCII space and removes all whitespace for `input_text`. Identical inputs are kept in one split. If an identical space-free input has different gold targets, every occurrence is excluded and recorded in `conflicts.csv`, because the distinction is impossible to learn from the available input.

The run writes `manifest.csv`, `train.csv`, `dev.csv`, `test.csv`, `conflicts.csv`, and `summary.json`.

## 2. Train

Replace `<prep-run>` in `./logs/prep/yonghe_qiang_01/<prep-run>/...` from the `config/training/train_yq.yaml`, then run:

```bash
python -m src.training.train_boundary --config config/training/train_yq.yaml
```

The default two-layer bidirectional LSTM uses grapheme embeddings and a weighted binary boundary loss. It chooses the checkpoint and insertion threshold by development boundary F1 with early stopping. Each external model run contains:

- `model.pt`, `model_config.json`, and `vocab.json`
- `threshold.json` and the resolved `run_config.json`
- `train_log.tsv` and `test_metrics.json`

Metrics include boundary precision/recall/F1, exact sentence accuracy, and word error rate.

## 3. Restore files

Edit `config/inference/restore_yq.yaml` to select the space-free sources and a trained model run. Validate first, then write timestamped copies:

```bash
python -m src.inference.restore_spaces --config config/inference/restore_yq.yaml --dry-run
python -m src.inference.restore_spaces --config config/inference/restore_yq.yaml
```

Only selected annotation values are replaced. EAF tier and annotation IDs, TextGrid intervals and points, table cells outside selected columns, and all unselected metadata remain in their original positions. Existing whitespace in selected values is removed before prediction, making the command suitable for both fully space-free and partially spaced ASR output.

Inference logs `predictions.csv` with the source path, tier/column locator, original text, model input, and restored text, plus `summary.json`.
