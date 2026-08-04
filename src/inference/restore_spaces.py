from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config import load_config, resolve_path, timestamp
from src.data.formats import SourceFile, copy_unchanged, discover_sources, read_annotations, write_replacements
from src.training.model import BoundaryBiLSTM, BoundaryModelConfig
from src.training.text import graphemes, reconstruct, space_free


@dataclass(frozen=True)
class PendingAnnotation:
    source: SourceFile
    locator: str
    container: str
    original_text: str
    input_text: str


class BoundaryPredictor:
    def __init__(self, model_dir: Path, device: torch.device):
        self.device = device
        self.vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        model_config = BoundaryModelConfig(**json.loads((model_dir / "model_config.json").read_text(encoding="utf-8")))
        self.model = BoundaryBiLSTM(model_config).to(device)
        state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = float(json.loads((model_dir / "threshold.json").read_text())["threshold"])

    def predict(self, texts: list[str], batch_size: int = 128) -> list[str]:
        outputs: list[str] = []
        unknown = self.vocab["<UNK>"]
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                clusters = [graphemes(text) for text in batch_texts]
                nonempty = [items for items in clusters if items]
                if not nonempty:
                    outputs.extend("" for _ in batch_texts)
                    continue
                lengths = torch.tensor([len(items) for items in clusters], dtype=torch.long)
                max_length = int(lengths.max())
                input_ids = torch.zeros((len(clusters), max_length), dtype=torch.long)
                for row, items in enumerate(clusters):
                    input_ids[row, :len(items)] = torch.tensor([self.vocab.get(item, unknown) for item in items])
                logits = self.model(input_ids.to(self.device), lengths)
                probabilities = torch.sigmoid(logits).cpu()
                for row, items in enumerate(clusters):
                    boundaries = [value >= self.threshold for value in probabilities[row, :len(items)].tolist()]
                    if boundaries:
                        boundaries[-1] = False
                    outputs.append(reconstruct(items, boundaries))
        return outputs


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def restore(config_path: str | Path, *, dry_run: bool = False, run_timestamp: str | None = None):
    config, project_dir = load_config(config_path)
    sources = discover_sources(config, project_dir)
    model_dir = resolve_path(config["model_dir"], project_dir)
    predictor = BoundaryPredictor(model_dir, _device(str(config.get("device", "auto"))))
    pending: list[PendingAnnotation] = []
    source_annotations: dict[Path, list] = {}
    for source in sources:
        annotations = read_annotations(source, config)
        source_annotations[source.path] = annotations
        for annotation in annotations:
            input_text = space_free(annotation.text)
            if input_text:
                pending.append(PendingAnnotation(source, annotation.locator, annotation.container, annotation.text, input_text))
    restored = predictor.predict([item.input_text for item in pending], int(config.get("batch_size", 128)))
    rows = []
    replacements_by_source: dict[Path, dict[str, str]] = {source.path: {} for source in sources}
    for item, output in zip(pending, restored):
        if space_free(output) != item.input_text:
            raise AssertionError(f"Character-preservation invariant failed at {item.source.path}:{item.locator}")
        replacements_by_source[item.source.path][item.locator] = output
        rows.append({
            "data_root": str(item.source.data_root),
            "source_path": item.source.relative_path.as_posix(),
            "source_type": item.source.source_type,
            "locator": item.locator,
            "container": item.container,
            "original_text": item.original_text,
            "input_text": item.input_text,
            "restored_text": output,
            "changed": str(output != item.original_text).lower(),
        })

    run_timestamp = run_timestamp or timestamp()
    output_root = resolve_path(config["output_root"], project_dir) / run_timestamp
    logs_root = resolve_path(config.get("logs_dir", "logs/inference"), project_dir)
    language = str(config.get("language", "language"))
    log_dir = logs_root / language / run_timestamp
    summary = {
        "run_timestamp": run_timestamp,
        "model_dir": str(model_dir),
        "source_files": len(sources),
        "annotations_selected": sum(len(value) for value in source_annotations.values()),
        "annotations_predicted": len(pending),
        "annotations_changed": sum(row["changed"] == "true" for row in rows),
        "output_root": str(output_root),
    }
    if dry_run:
        return summary, output_root, log_dir
    existing = [path for path in (output_root, log_dir) if path.exists()]
    if existing:
        raise FileExistsError("Timestamped inference destination exists: " + ", ".join(map(str, existing)))
    output_root.mkdir(parents=True)
    roots = list(dict.fromkeys(source.data_root for source in sources))
    for source in sources:
        prefix = Path() if len(roots) == 1 else Path(f"root-{roots.index(source.data_root)}-{source.data_root.name}")
        destination = output_root / prefix / source.relative_path
        replacements = replacements_by_source[source.path]
        if replacements:
            write_replacements(source, destination, config, replacements)
        else:
            copy_unchanged(source, destination)
    log_dir.mkdir(parents=True)
    fields = ["data_root", "source_path", "source_type", "locator", "container", "original_text", "input_text", "restored_text", "changed"]
    with (log_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, output_root, log_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore spaces in selected transcription annotations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary, output_root, log_dir = restore(args.config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print(f"Would write files: {output_root}")
        print(f"Would write logs: {log_dir}")
    else:
        print(f"Wrote files: {output_root}")
        print(f"Wrote logs: {log_dir}")


if __name__ == "__main__":
    main()
