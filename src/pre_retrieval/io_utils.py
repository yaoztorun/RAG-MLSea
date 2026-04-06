from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TypeVar
from urllib.parse import quote, unquote


GRAPHDB_WRAPPER_PREFIX = "http://localhost:7200/resource?uri="
T = TypeVar("T")


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> None:
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        cleaned = normalize_whitespace(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def truncate_text(value: Any, max_characters: int) -> str:
    text = normalize_whitespace(value)
    if max_characters <= 0 or len(text) <= max_characters:
        return text

    candidate = text[: max_characters - 1].rstrip()
    if " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0].rstrip()
    return f"{candidate}…" if candidate else text[:max_characters]


def chunked(values: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    for index in range(0, len(values), chunk_size):
        yield values[index : index + chunk_size]


def fully_unquote(value: str) -> str:
    previous = value.strip()
    current = unquote(previous)
    while current != previous:
        previous = current
        current = unquote(previous)
    return current.strip()


def normalize_identifier(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(GRAPHDB_WRAPPER_PREFIX):
        cleaned = cleaned[len(GRAPHDB_WRAPPER_PREFIX) :]
    return fully_unquote(cleaned)


def build_document_id(representation_type: str, paper_id: str) -> str:
    return f"{representation_type}::{quote(paper_id, safe='')}"
