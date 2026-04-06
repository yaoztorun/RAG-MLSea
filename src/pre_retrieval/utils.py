from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TypeVar
from urllib.parse import quote, unquote

import numpy as np


GRAPHDB_WRAPPER_PREFIX = "http://localhost:7200/resource?uri="
DEFAULT_RDF_INPUT_PATH = "data/raw/pwc_1.nt"
DEFAULT_SAMPLE_RDF_INPUT_PATH = "data/raw/pwc_1_sample.nt"
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


def approx_token_count(value: str) -> int:
    text = normalize_whitespace(value)
    return len(text.split()) if text else 0


def compute_distribution_stats(values: Sequence[int | float]) -> Dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(mean(array.tolist())),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


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


def paper_id_from_uri(paper_uri: str) -> str:
    return normalize_identifier(paper_uri)


def build_item_id(representation_type: str, paper_id: str) -> str:
    return f"{representation_type}::{quote(paper_id, safe='')}"


def collection_name_for_representation(representation_type: str) -> str:
    return f"papers_{representation_type}"


def missing_input_message(path: Path) -> str:
    if path.suffix == ".nt":
        return (
            f"Input file not found: {path}\n"
            f"Place the full RDF dump at {DEFAULT_RDF_INPUT_PATH} or use the optional sample file "
            f"{DEFAULT_SAMPLE_RDF_INPUT_PATH} with --input-path."
        )
    return (
        f"Input file not found: {path}\n"
        "Check the path or rebuild the required upstream artifact before rerunning this step."
    )


def require_existing_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(missing_input_message(path))
