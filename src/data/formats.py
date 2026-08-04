from __future__ import annotations

import csv
import io
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import require_list, resolve_path, selector


SUPPORTED_FORMATS = {".eaf", ".textgrid", ".csv", ".tsv", ".txt"}
GENERATED_DIRS = {"space-recognised", "space-recognized"}
TEXTGRID_ITEM_RE = re.compile(r"^\s*item\s+\[\d+\]:\s*$")
TEXTGRID_CLASS_RE = re.compile(r'^\s*class\s*=\s*"((?:""|[^"])*)"\s*$')
TEXTGRID_NAME_RE = re.compile(r'^\s*name\s*=\s*"((?:""|[^"])*)"\s*$')
TEXTGRID_VALUE_RE = re.compile(
    r'^(\s*(?:text|mark)\s*=\s*")((?:""|[^"])*)("\s*)(\r?\n)?$'
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    data_root: Path
    relative_path: Path
    source_type: str


@dataclass(frozen=True)
class Annotation:
    locator: str
    container: str
    text: str


@dataclass(frozen=True)
class EncodedText:
    text: str
    encoding: str
    bom: bytes


def canonical_format(value: str) -> str:
    value = value.strip()
    if not value.startswith("."):
        value = "." + value
    value = value.casefold()
    if value not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported transcription format: {value}")
    return value


def data_roots(config: dict[str, Any], project_dir: Path) -> list[Path]:
    roots = [resolve_path(item, project_dir) for item in require_list(config, "data_roots", [])]
    if not roots:
        raise ValueError("data_roots must contain at least one directory")
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise ValueError("Configured data root is not a directory: " + ", ".join(missing))
    return roots


def discover_sources(config: dict[str, Any], project_dir: Path) -> list[SourceFile]:
    roots = data_roots(config, project_dir)
    patterns = require_list(config, "transcription_globs", ["**/*"])
    selected = {canonical_format(item) for item in require_list(config, "prefer", [])}
    if not selected:
        raise ValueError("prefer must select at least one transcription format")
    sources: list[SourceFile] = []
    seen: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            candidates = root.glob(pattern)
            if "**" not in pattern:
                candidates = (*candidates, *root.rglob(pattern))
            for path in candidates:
                if not path.is_file() or path.resolve() in seen:
                    continue
                relative = path.relative_to(root)
                if any(part.casefold() in GENERATED_DIRS for part in relative.parts[:-1]):
                    continue
                source_type = path.suffix.casefold()
                if source_type not in selected:
                    continue
                seen.add(path.resolve())
                sources.append(SourceFile(path.resolve(), root, relative, source_type))
    sources.sort(key=lambda item: (str(item.data_root), item.relative_path.as_posix()))
    if not sources:
        raise ValueError("No transcription files matched the configured roots, globs, and formats")
    return sources


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _select_names(available: list[str], requested: list[str], path: Path, kind: str) -> list[str]:
    if requested[0].casefold() == "all":
        return available
    chosen: list[str] = []
    for name in requested:
        matches = [item for item in available if item.casefold() == name.casefold()]
        if not matches:
            rendered = ", ".join(repr(item) for item in available) or "<none>"
            raise ValueError(f"{path}: requested {kind} {name!r} not found; available: {rendered}")
        if len(matches) > 1:
            raise ValueError(f"{path}: requested {kind} {name!r} is ambiguous")
        chosen.append(matches[0])
    return chosen


def _read_encoded(path: Path) -> EncodedText:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return EncodedText(raw[2:].decode("utf-16-le"), "utf-16-le", b"\xff\xfe")
    if raw.startswith(b"\xfe\xff"):
        return EncodedText(raw[2:].decode("utf-16-be"), "utf-16-be", b"\xfe\xff")
    if raw.startswith(b"\xef\xbb\xbf"):
        return EncodedText(raw[3:].decode("utf-8"), "utf-8", b"\xef\xbb\xbf")
    return EncodedText(raw.decode("utf-8"), "utf-8", b"")


def _write_encoded(path: Path, content: str, encoded: EncodedText) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.bom + content.encode(encoded.encoding))


def _table_settings(config: dict[str, Any], source_type: str) -> tuple[str, bool, list[str]]:
    name = source_type.lstrip(".")
    key = f"{name}_text_column"
    if key not in config:
        raise ValueError(f"{key} must be configured when selecting {source_type}")
    selectors = selector(config, key)
    has_header = bool(config.get(f"{name}_has_header", source_type == ".csv"))
    if source_type == ".csv":
        delimiter = ","
    elif source_type == ".tsv":
        delimiter = "\t"
    else:
        raw = str(config.get("txt_delimiter", "")).strip()
        delimiter = {"tab": "\t", "comma": ","}.get(raw.casefold(), raw)
        if len(delimiter) != 1:
            raise ValueError("txt_delimiter must be 'tab', 'comma', or one literal character")
    return delimiter, has_header, selectors


def _column_indices(selectors: list[str], header: list[str] | None, width: int, path: Path) -> list[int]:
    if selectors[0].casefold() == "all":
        return list(range(width))
    indices: set[int] = set()
    for requested in selectors:
        if requested.isdecimal():
            index = int(requested) - 1
            if not 0 <= index < width:
                raise ValueError(f"{path}: column {requested} is outside 1..{width}")
            indices.add(index)
            continue
        if header is None:
            raise ValueError(f"{path}: headerless tables require numeric column selectors")
        matches = [i for i, name in enumerate(header) if name.casefold() == requested.casefold()]
        if not matches:
            raise ValueError(f"{path}: requested column {requested!r} not found")
        if len(matches) > 1:
            raise ValueError(f"{path}: requested column {requested!r} is ambiguous")
        indices.add(matches[0])
    return sorted(indices)


def read_annotations(source: SourceFile, config: dict[str, Any]) -> list[Annotation]:
    if source.source_type == ".eaf":
        return _read_eaf(source.path, config)
    if source.source_type == ".textgrid":
        return _read_textgrid(source.path, config)
    return _read_table(source.path, source.source_type, config)


def _read_eaf(path: Path, config: dict[str, Any]) -> list[Annotation]:
    root = ET.parse(path).getroot()
    tiers = [element for element in root.iter() if _strip_namespace(element.tag) == "TIER"]
    chosen = set(_select_names([tier.get("TIER_ID", "") for tier in tiers], selector(config, "eaf_tier"), path, "EAF tier"))
    records: list[Annotation] = []
    for tier in tiers:
        tier_id = tier.get("TIER_ID", "")
        if tier_id not in chosen:
            continue
        ordinal = 0
        for annotation in tier.iter():
            if _strip_namespace(annotation.tag) not in {"ALIGNABLE_ANNOTATION", "REF_ANNOTATION"}:
                continue
            value = next((child for child in annotation.iter() if _strip_namespace(child.tag) == "ANNOTATION_VALUE"), None)
            if value is None:
                continue
            annotation_id = annotation.get("ANNOTATION_ID") or str(ordinal)
            records.append(Annotation(f"eaf:{tier_id}:{annotation_id}", tier_id, value.text or ""))
            ordinal += 1
    return records


def _textgrid_parts(path: Path) -> tuple[EncodedText, list[str], list[tuple[str, int, int]]]:
    encoded = _read_encoded(path)
    lines = encoded.text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if TEXTGRID_ITEM_RE.match(line.rstrip("\r\n"))]
    if not starts:
        raise ValueError(f"{path}: expected Praat long text-format tier blocks")
    tiers: list[tuple[str, int, int]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        tier_class = name = ""
        for line in lines[start:end]:
            bare = line.rstrip("\r\n")
            class_match = TEXTGRID_CLASS_RE.match(bare)
            name_match = TEXTGRID_NAME_RE.match(bare)
            if class_match and not tier_class:
                tier_class = class_match.group(1).replace('""', '"')
            if name_match and not name:
                name = name_match.group(1).replace('""', '"')
        if tier_class not in {"IntervalTier", "TextTier"} or not name:
            raise ValueError(f"{path}: malformed TextGrid tier at line {start + 1}")
        tiers.append((name, start, end))
    return encoded, lines, tiers


def _read_textgrid(path: Path, config: dict[str, Any]) -> list[Annotation]:
    _, lines, tiers = _textgrid_parts(path)
    chosen = set(_select_names([name for name, _, _ in tiers], selector(config, "textgrid_tier"), path, "TextGrid tier"))
    records: list[Annotation] = []
    for name, start, end in tiers:
        if name not in chosen:
            continue
        ordinal = 0
        for index in range(start, end):
            match = TEXTGRID_VALUE_RE.match(lines[index])
            if match:
                records.append(Annotation(f"textgrid:{name}:{ordinal}", name, match.group(2).replace('""', '"')))
                ordinal += 1
    return records


def _table_parts(path: Path, source_type: str, config: dict[str, Any]):
    delimiter, has_header, selectors = _table_settings(config, source_type)
    encoded = _read_encoded(path)
    rows = list(csv.reader(io.StringIO(encoded.text, newline=""), delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: table is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{path}: rows have inconsistent column counts")
    header = rows[0] if has_header else None
    indices = _column_indices(selectors, header, width, path)
    return encoded, rows, delimiter, has_header, indices


def _read_table(path: Path, source_type: str, config: dict[str, Any]) -> list[Annotation]:
    _, rows, _, has_header, indices = _table_parts(path, source_type, config)
    first = 1 if has_header else 0
    records: list[Annotation] = []
    for row_index in range(first, len(rows)):
        for column_index in indices:
            records.append(Annotation(f"table:{row_index}:{column_index}", str(column_index + 1), rows[row_index][column_index]))
    return records


def write_replacements(source: SourceFile, destination: Path, config: dict[str, Any], replacements: dict[str, str]) -> None:
    if source.source_type == ".eaf":
        _write_eaf(source.path, destination, config, replacements)
    elif source.source_type == ".textgrid":
        _write_textgrid(source.path, destination, config, replacements)
    else:
        _write_table(source.path, destination, source.source_type, config, replacements)


def _write_eaf(path: Path, destination: Path, config: dict[str, Any], replacements: dict[str, str]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    tiers = [element for element in root.iter() if _strip_namespace(element.tag) == "TIER"]
    chosen = set(_select_names([tier.get("TIER_ID", "") for tier in tiers], selector(config, "eaf_tier"), path, "EAF tier"))
    for tier in tiers:
        tier_id = tier.get("TIER_ID", "")
        if tier_id not in chosen:
            continue
        ordinal = 0
        for annotation in tier.iter():
            if _strip_namespace(annotation.tag) not in {"ALIGNABLE_ANNOTATION", "REF_ANNOTATION"}:
                continue
            value = next((child for child in annotation.iter() if _strip_namespace(child.tag) == "ANNOTATION_VALUE"), None)
            if value is None:
                continue
            annotation_id = annotation.get("ANNOTATION_ID") or str(ordinal)
            locator = f"eaf:{tier_id}:{annotation_id}"
            if locator in replacements:
                value.text = replacements[locator]
            ordinal += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree.write(destination, encoding="UTF-8", xml_declaration=True)


def _write_textgrid(path: Path, destination: Path, config: dict[str, Any], replacements: dict[str, str]) -> None:
    encoded, lines, tiers = _textgrid_parts(path)
    chosen = set(_select_names([name for name, _, _ in tiers], selector(config, "textgrid_tier"), path, "TextGrid tier"))
    for name, start, end in tiers:
        if name not in chosen:
            continue
        ordinal = 0
        for index in range(start, end):
            match = TEXTGRID_VALUE_RE.match(lines[index])
            if not match:
                continue
            locator = f"textgrid:{name}:{ordinal}"
            if locator in replacements:
                escaped = replacements[locator].replace('"', '""')
                lines[index] = match.group(1) + escaped + match.group(3) + (match.group(4) or "")
            ordinal += 1
    _write_encoded(destination, "".join(lines), encoded)


def _write_table(path: Path, destination: Path, source_type: str, config: dict[str, Any], replacements: dict[str, str]) -> None:
    encoded, rows, delimiter, _, indices = _table_parts(path, source_type, config)
    for row_index in range(len(rows)):
        for column_index in indices:
            locator = f"table:{row_index}:{column_index}"
            if locator in replacements:
                rows[row_index][column_index] = replacements[locator]
    newline = "\r\n" if "\r\n" in encoded.text else "\n"
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=delimiter, lineterminator=newline).writerows(rows)
    _write_encoded(destination, buffer.getvalue(), encoded)


def copy_unchanged(source: SourceFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.path, destination)
