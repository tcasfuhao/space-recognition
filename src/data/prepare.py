from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_path, timestamp
from src.data.formats import Annotation, SourceFile, discover_sources, read_annotations


WS_RE = re.compile(r"\s+")
MANIFEST_FIELDS = [
    "example_id", "data_root", "source_path", "source_type", "locator",
    "container", "input_text", "gold_text", "split",
]


@dataclass
class Example:
    example_id: str
    data_root: str
    source_path: str
    source_type: str
    locator: str
    container: str
    input_text: str
    gold_text: str
    split: str = ""


def canonical_gold(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def remove_whitespace(text: str) -> str:
    return WS_RE.sub("", text)


def make_example(source: SourceFile, annotation: Annotation) -> Example | None:
    gold = canonical_gold(annotation.text)
    if not gold:
        return None
    input_text = remove_whitespace(gold)
    identity = f"{source.data_root}\0{source.relative_path.as_posix()}\0{annotation.locator}"
    example_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return Example(
        example_id=example_id,
        data_root=str(source.data_root),
        source_path=source.relative_path.as_posix(),
        source_type=source.source_type,
        locator=annotation.locator,
        container=annotation.container,
        input_text=input_text,
        gold_text=gold,
    )


def collect_examples(config: dict[str, Any], project_dir: Path):
    sources = discover_sources(config, project_dir)
    examples: list[Example] = []
    empty_count = 0
    for source in sources:
        annotations = read_annotations(source, config)
        for annotation in annotations:
            example = make_example(source, annotation)
            if example is None:
                empty_count += 1
            else:
                examples.append(example)
    return sources, examples, empty_count


def filter_conflicts(examples: list[Example]):
    targets: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        targets[example.input_text].add(example.gold_text)
    conflicting_inputs = {text for text, values in targets.items() if len(values) > 1}
    kept = [example for example in examples if example.input_text not in conflicting_inputs]
    conflicts = [example for example in examples if example.input_text in conflicting_inputs]
    return kept, conflicts


def assign_grouped_splits(
    examples: list[Example], *, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float
) -> None:
    ratios = [train_ratio, dev_ratio, test_ratio]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train_ratio, dev_ratio, and test_ratio must be non-negative and sum to 1")
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.input_text].append(example)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(groups[key]) for key in keys)
    targets = {"train": total * train_ratio, "dev": total * dev_ratio, "test": total * test_ratio}
    counts = {name: 0 for name in targets}
    names = ["train", "dev", "test"]
    for key in keys:
        size = len(groups[key])
        candidates = [name for name, ratio in zip(names, ratios) if ratio > 0]
        split = max(candidates, key=lambda name: (targets[name] - counts[name], -names.index(name)))
        for example in groups[key]:
            example.split = split
        counts[split] += size


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path: str | Path, *, dry_run: bool = False, run_timestamp: str | None = None):
    config, project_dir = load_config(config_path)
    sources, examples, empty_count = collect_examples(config, project_dir)
    kept, conflicts = filter_conflicts(examples)
    assign_grouped_splits(
        kept,
        seed=int(config.get("split_seed", 42)),
        train_ratio=float(config.get("train_ratio", 0.8)),
        dev_ratio=float(config.get("dev_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
    )
    run_timestamp = run_timestamp or timestamp()
    logs_root = resolve_path(config.get("logs_dir", "logs/prep"), project_dir)
    language = str(config.get("language", "language"))
    if Path(language).name != language or language in {"", ".", ".."}:
        raise ValueError("language must be one non-empty path segment")
    run_dir = logs_root / language / run_timestamp
    split_counts = {name: sum(item.split == name for item in kept) for name in ("train", "dev", "test")}
    summary = {
        "run_timestamp": run_timestamp,
        "source_files": len(sources),
        "annotations_nonempty": len(examples),
        "annotations_empty": empty_count,
        "annotations_excluded_conflict": len(conflicts),
        "conflicting_input_forms": len({item.input_text for item in conflicts}),
        "annotations_kept": len(kept),
        "unique_kept_inputs": len({item.input_text for item in kept}),
        **{f"{name}_rows": count for name, count in split_counts.items()},
    }
    if dry_run:
        return summary, run_dir
    if run_dir.exists():
        raise FileExistsError(f"Timestamped preparation destination exists: {run_dir}")
    run_dir.mkdir(parents=True)
    rows = [asdict(item) for item in kept]
    _write_csv(run_dir / "manifest.csv", rows, MANIFEST_FIELDS)
    for split in ("train", "dev", "test"):
        _write_csv(run_dir / f"{split}.csv", [row for row in rows if row["split"] == split], MANIFEST_FIELDS)
    conflict_fields = MANIFEST_FIELDS[:-1] + ["competing_targets"]
    conflict_rows = []
    target_map: dict[str, set[str]] = defaultdict(set)
    for item in conflicts:
        target_map[item.input_text].add(item.gold_text)
    for item in conflicts:
        row = asdict(item)
        row.pop("split", None)
        row["competing_targets"] = json.dumps(sorted(target_map[item.input_text]), ensure_ascii=False)
        conflict_rows.append(row)
    _write_csv(run_dir / "conflicts.csv", conflict_rows, conflict_fields)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gold spacing and create grouped train/dev/test manifests.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary, run_dir = prepare(args.config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'Would write' if args.dry_run else 'Wrote'}: {run_dir}")


if __name__ == "__main__":
    main()

