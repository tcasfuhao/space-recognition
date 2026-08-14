from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import load_config, model_timestamp, resolve_path
from src.training.metrics import BoundaryMetrics
from src.training.model import BoundaryBiLSTM, BoundaryModelConfig
from src.training.text import gold_clusters_and_labels, graphemes, reconstruct


PAD = "<PAD>"
UNK = "<UNK>"
SPLIT_NAMES = ("train", "dev", "test")
OBSOLETE_SPLIT_KEYS = {"train_csv", "dev_csv", "test_csv"}
NORMALISATION_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class TextExample:
    input_text: str
    gold_text: str
    data_root: str = ""


class BoundaryDataset(Dataset):
    def __init__(self, examples: list[TextExample], vocab: dict[str, int]):
        self.items = []
        for example in examples:
            clusters, labels = gold_clusters_and_labels(example.gold_text)
            if "".join(clusters) != example.input_text:
                raise ValueError(f"Gold text does not reduce to input text: {example.gold_text!r}")
            if not clusters:
                continue
            ids = [vocab.get(cluster, vocab[UNK]) for cluster in clusters]
            self.items.append((ids, labels, clusters, example.gold_text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    max_length = max(len(item[0]) for item in items)
    input_ids = torch.zeros((len(items), max_length), dtype=torch.long)
    labels = torch.zeros((len(items), max_length), dtype=torch.float32)
    mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    lengths = torch.tensor([len(item[0]) for item in items], dtype=torch.long)
    clusters = []
    gold_texts = []
    for row, (ids, item_labels, item_clusters, gold_text) in enumerate(items):
        size = len(ids)
        input_ids[row, :size] = torch.tensor(ids)
        labels[row, :size] = torch.tensor(item_labels)
        mask[row, :size] = True
        clusters.append(item_clusters)
        gold_texts.append(gold_text)
    return {"input_ids": input_ids, "labels": labels, "mask": mask, "lengths": lengths, "clusters": clusters, "gold_texts": gold_texts}


def read_examples(path: Path) -> list[TextExample]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"input_text", "gold_text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain input_text and gold_text columns")
        return [
            TextExample(row["input_text"], row["gold_text"], row.get("data_root", ""))
            for row in reader
            if row["input_text"]
        ]


def resolve_split_paths(config: dict[str, Any], project_dir: Path) -> tuple[Path, dict[str, Path]]:
    obsolete = sorted(OBSOLETE_SPLIT_KEYS.intersection(config))
    if obsolete:
        raise ValueError(
            f"Obsolete training config keys: {', '.join(obsolete)}; use splits_dir instead"
        )
    if not config.get("splits_dir"):
        raise ValueError("splits_dir must point to one exact preparation run")
    splits_dir = resolve_path(config["splits_dir"], project_dir)
    if not splits_dir.is_dir():
        raise ValueError(f"splits_dir is not a directory: {splits_dir}")
    paths = {name: splits_dir / f"{name}.csv" for name in SPLIT_NAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Missing generated split manifest(s): " + ", ".join(missing))
    return splits_dir, paths


def resolve_output_root(
    config: dict[str, Any],
    project_dir: Path,
    examples_by_split: dict[str, list[TextExample]],
) -> Path:
    if config.get("out_dir"):
        return resolve_path(config["out_dir"], project_dir)

    examples = [item for values in examples_by_split.values() for item in values]
    if any(not item.data_root for item in examples):
        raise ValueError(
            "Generated manifests must contain data_root when out_dir is not configured"
        )
    normalisation_roots = {
        resolve_path(item.data_root, project_dir) for item in examples
    }
    invalid = [
        root
        for root in normalisation_roots
        if root.parent.name.casefold() != "normalised"
        or not NORMALISATION_TIMESTAMP_RE.fullmatch(root.name)
    ]
    if invalid:
        raise ValueError(
            "Cannot derive out_dir; data_root must use <dataset-root>/normalised/<timestamp>: "
            + ", ".join(map(str, sorted(invalid)))
        )
    dataset_roots = {root.parent.parent for root in normalisation_roots}
    if len(dataset_roots) != 1:
        raise ValueError(
            "Manifests contain multiple dataset roots; configure out_dir explicitly"
        )
    return next(iter(dataset_roots)) / "processed" / "spaces"


def make_vocab(examples: list[TextExample]) -> dict[str, int]:
    symbols = sorted({cluster for item in examples for cluster in graphemes(item.input_text)})
    return {PAD: 0, UNK: 1, **{symbol: index + 2 for index, symbol in enumerate(symbols)}}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def predict_probabilities(model, loader, device):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["lengths"])
            probabilities = torch.sigmoid(logits).cpu()
            for row, length in enumerate(batch["lengths"].tolist()):
                output.append((probabilities[row, :length].tolist(), batch["labels"][row, :length].int().tolist(), batch["clusters"][row], batch["gold_texts"][row]))
    return output


def score_predictions(predictions, threshold: float) -> dict[str, float | int]:
    metrics = BoundaryMetrics()
    for probabilities, labels, clusters, gold_text in predictions:
        predicted = [int(value >= threshold) for value in probabilities]
        if predicted:
            predicted[-1] = 0
        predicted_text = reconstruct(clusters, predicted)
        metrics.update(predicted, labels, predicted_text, gold_text)
        if "".join(predicted_text.split()) != "".join(clusters):
            raise AssertionError("Character-preservation invariant failed")
    return metrics.as_dict()


def best_threshold(predictions) -> tuple[float, dict[str, float | int]]:
    candidates = [index / 100 for index in range(5, 100, 5)]
    scored = [(threshold, score_predictions(predictions, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["boundary_f1"], -abs(item[0] - 0.5)))


def masked_loss(logits, labels, mask, positive_weight):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=positive_weight)
    return losses.masked_select(mask).mean()


def train(config_path: str | Path):
    config, project_dir = load_config(config_path)
    splits_dir, split_paths = resolve_split_paths(config, project_dir)
    examples_by_split = {
        name: read_examples(path) for name, path in split_paths.items()
    }
    empty = [name for name, examples in examples_by_split.items() if not examples]
    if empty:
        raise ValueError("Split manifest(s) contain no examples: " + ", ".join(empty))
    train_examples = examples_by_split["train"]
    dev_examples = examples_by_split["dev"]
    test_examples = examples_by_split["test"]
    out_root = resolve_output_root(config, project_dir, examples_by_split)

    seed = int(config.get("train_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(str(config.get("device", "auto")))
    vocab = make_vocab(train_examples)
    model_config = BoundaryModelConfig(
        vocab_size=len(vocab),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 256)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model = BoundaryBiLSTM(model_config).to(device)
    train_dataset = BoundaryDataset(train_examples, vocab)
    dev_dataset = BoundaryDataset(dev_examples, vocab)
    test_dataset = BoundaryDataset(test_examples, vocab)
    batch_size = int(config.get("batch_size", 64))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)

    positives = sum(sum(item[1]) for item in train_dataset.items)
    total = sum(len(item[1]) for item in train_dataset.items)
    positive_weight = torch.tensor((total - positives) / max(positives, 1), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 7))
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    best_dev_threshold = 0.5
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["lengths"])
            loss = masked_loss(logits, batch["labels"].to(device), batch["mask"].to(device), positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            losses.append(loss.item())
        dev_predictions = predict_probabilities(model, dev_loader, device)
        threshold, dev_metrics = best_threshold(dev_predictions)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "threshold": threshold, **dev_metrics}
        history.append(row)
        print(json.dumps(row))
        f1 = float(dev_metrics["boundary_f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_dev_threshold = threshold
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = score_predictions(predict_probabilities(model, test_loader, device), best_dev_threshold)

    model_name = str(config.get("model_name", "bilstm-boundary"))
    run_dir = out_root / f"{model_name}_{model_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, run_dir / "model.pt")
    (run_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "model_config.json").write_text(json.dumps(model.serializable_config(), indent=2), encoding="utf-8")
    (run_dir / "threshold.json").write_text(json.dumps({"threshold": best_dev_threshold, "best_epoch": best_epoch, "best_dev_boundary_f1": best_f1}, indent=2), encoding="utf-8")
    run_config = {
        **config,
        "resolved_splits_dir": str(splits_dir),
        **{f"resolved_{name}_csv": str(path) for name, path in split_paths.items()},
        "resolved_out_dir": str(out_root),
        "device_used": str(device),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    with (run_dir / "train_log.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    print(f"Wrote model run: {run_dir}")
    print(json.dumps(test_metrics, indent=2))
    return run_dir, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the grapheme boundary BiLSTM.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
