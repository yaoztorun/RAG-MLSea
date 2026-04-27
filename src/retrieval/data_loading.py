from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from src.retrieval.config import (
    ENTITY_TYPE_FOLDERS,
    PRE_RETRIEVAL_BASE_PATHS,
    QUESTIONS_PATH,
)

MLSEA_PREFIX = "http://w3id.org/mlsea/pwc/"
GRAPHDB_WRAPPER_PREFIX = "http://localhost:7200/resource?uri="

_GOLD_ID_KEYS = [
    "gold_paper_id",
    "gold_dataset_id",
    "gold_model_id",
    "target_entity_iri",
    "gold_entity_id",
]
_ENTITY_ID_KEYS = ["paper_id", "dataset_id", "model_id", "entity_id"]
_CANDIDATE_LIST_KEYS = ["documents", "retrieved", "candidates", "top_k_results", "ranked_results"]
_ENTRY_LIST_KEYS = ["entries", "results", "questions", "per_question_results"]


def normalize_iri(iri: str) -> str:
    if not iri:
        return ""
    cleaned = iri.strip()
    if cleaned.startswith(GRAPHDB_WRAPPER_PREFIX):
        cleaned = cleaned[len(GRAPHDB_WRAPPER_PREFIX):]
    previous = cleaned
    current = unquote(previous)
    while current != previous:
        previous = current
        current = unquote(previous)
    return current.strip()


def entity_type_from_iri(iri: str) -> str:
    normalized = normalize_iri(iri)
    if "/scientificWork/" in normalized:
        return "paper"
    if "/dataset/" in normalized:
        return "dataset"
    if "/model/" in normalized:
        return "model"
    return "unknown"


def find_top10_path(entity_type: str, representation: str) -> Path | None:
    folder = ENTITY_TYPE_FOLDERS.get(entity_type, f"{entity_type}_results")
    for base in PRE_RETRIEVAL_BASE_PATHS:
        candidate = base / folder / representation / "top10.json"
        if candidate.exists():
            return candidate
    return None


def find_results_path(entity_type: str, representation: str) -> Path | None:
    folder = ENTITY_TYPE_FOLDERS.get(entity_type, f"{entity_type}_results")
    for base in PRE_RETRIEVAL_BASE_PATHS:
        candidate = base / folder / representation / "results.json"
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _extract_gold_id(entry: dict[str, Any]) -> str:
    for key in _GOLD_ID_KEYS:
        val = entry.get(key)
        if val:
            return str(val)
    return ""


def _extract_entity_id(doc: dict[str, Any]) -> str:
    for key in _ENTITY_ID_KEYS:
        val = doc.get(key)
        if val:
            return str(val)
    item_id = doc.get("item_id", "")
    if "::" in item_id:
        encoded = item_id.split("::", 1)[1]
        return normalize_iri(encoded)
    return item_id


def _extract_candidates_list(entry: dict[str, Any]) -> list[dict[str, Any]]:
    for key in _CANDIDATE_LIST_KEYS:
        val = entry.get(key)
        if val is not None:
            return val
    return []


def _extract_entries_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in _ENTRY_LIST_KEYS:
        val = data.get(key)
        if val is not None:
            return val
    return []


def _metadata_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "authors": doc.get("authors") or [],
        "publication_year": doc.get("publication_year"),
        "keywords": doc.get("keywords") or [],
        "tasks": doc.get("tasks") or [],
        "datasets": doc.get("datasets") or [],
        "methods": doc.get("methods") or [],
        "metrics": doc.get("metrics") or [],
        "implementations": doc.get("implementations") or [],
        "source_text": doc.get("source_text", ""),
    }


class PreRetrievalEntry:
    __slots__ = ("question_id", "question", "gold_entity_id", "raw_candidates")

    def __init__(
        self,
        question_id: str,
        question: str,
        gold_entity_id: str,
        raw_candidates: list[dict[str, Any]],
    ) -> None:
        self.question_id = question_id
        self.question = question
        self.gold_entity_id = normalize_iri(gold_entity_id)
        self.raw_candidates = raw_candidates


def load_top10(entity_type: str, representation: str) -> dict[str, PreRetrievalEntry]:
    path = find_top10_path(entity_type, representation)
    if path is None:
        return {}
    data = _load_json(path)
    entries = _extract_entries_list(data)
    index: dict[str, PreRetrievalEntry] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("question_id", "")).strip()
        if not qid:
            continue
        gold = _extract_gold_id(entry)
        docs = _extract_candidates_list(entry)
        index[qid] = PreRetrievalEntry(
            question_id=qid,
            question=str(entry.get("question", "")),
            gold_entity_id=gold,
            raw_candidates=docs,
        )
    return index


def build_raw_candidate(doc: dict[str, Any], representation_source: str) -> dict[str, Any]:
    entity_id = _extract_entity_id(doc)
    normalized = normalize_iri(entity_id)
    return {
        "entity_id": entity_id,
        "normalized_entity_id": normalized,
        "entity_type": entity_type_from_iri(normalized),
        "title_or_label": doc.get("title", "") or "",
        "original_rank": int(doc["rank"]) if doc.get("rank") is not None else None,
        "new_rank": int(doc["rank"]) if doc.get("rank") is not None else None,
        "original_score": float(doc["score"]) if doc.get("score") is not None else None,
        "method_score": float(doc["score"]) if doc.get("score") is not None else None,
        "representation_source": representation_source,
        "metadata": _metadata_from_doc(doc),
    }


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or QUESTIONS_PATH
    return _load_json(p)


def build_question_index(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(q.get("id", "")): q for q in questions if q.get("id")}
