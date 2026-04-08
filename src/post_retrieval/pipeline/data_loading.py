from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from src.pre_retrieval.config import REPO_ROOT, resolve_repo_path
from src.pre_retrieval.utils import load_json, load_jsonl, normalize_identifier

DEFAULT_CANONICAL_PAPERS_PATH = REPO_ROOT / "data/intermediate/raw_papers/papers_master.jsonl"
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data/questions/ml_questions_dataset.json"
DEFAULT_RETRIEVAL_RESULTS_DIR = REPO_ROOT / "data/retrieval_results"
DEFAULT_REPRESENTATIONS_DIR = REPO_ROOT / "data/intermediate/representations"


def _repo_path(path: str | Path) -> Path:
    return resolve_repo_path(path)


def resolve_retrieval_results_path(
    representation_type: str,
    results_dir: str | Path = DEFAULT_RETRIEVAL_RESULTS_DIR,
) -> Path:
    return _repo_path(results_dir) / f"{representation_type}_results.json"


def resolve_representation_path(
    representation_type: str,
    representations_dir: str | Path = DEFAULT_REPRESENTATIONS_DIR,
) -> Path:
    return _repo_path(representations_dir) / f"{representation_type}.jsonl"


def load_canonical_paper_records(
    path: str | Path = DEFAULT_CANONICAL_PAPERS_PATH,
) -> List[Dict[str, Any]]:
    return load_jsonl(_repo_path(path))


def build_paper_id_lookup(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        normalize_identifier(record.get("paper_id", "")): record
        for record in records
        if record.get("paper_id")
    }


def load_question_dataset(
    path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> List[Dict[str, Any]]:
    return load_json(_repo_path(path))


def build_question_id_lookup(questions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(question.get("id", "")): question
        for question in questions
        if question.get("id")
    }


def load_representation_records(
    representation_type: str,
    representations_dir: str | Path = DEFAULT_REPRESENTATIONS_DIR,
) -> List[Dict[str, Any]]:
    return load_jsonl(resolve_representation_path(representation_type, representations_dir))


RepresentationLookup = Dict[
    str,
    Union[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]],
]


def build_representation_lookup(records: Iterable[Dict[str, Any]]) -> RepresentationLookup:
    item_lookup: Dict[str, Dict[str, Any]] = {}
    paper_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        item_id = str(record.get("item_id", "")).strip()
        representation_type = str(record.get("representation_type", "")).strip()
        paper_id = normalize_identifier(record.get("paper_id", ""))
        if item_id:
            item_lookup[item_id] = record
        if representation_type and paper_id:
            paper_lookup[(representation_type, paper_id)] = record
    return {"item_id": item_lookup, "paper": paper_lookup}


def load_retrieval_payload(path: str | Path) -> Dict[str, Any]:
    return load_json(_repo_path(path))


def get_per_question_entries(payload: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    return list(payload.get("per_question", []))


def build_retrieval_question_lookup(
    payload: Dict[str, Any] | List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in get_per_question_entries(payload):
        question_id = str(entry.get("question_id", "")).strip()
        if question_id:
            lookup[question_id] = entry
    return lookup


def resolve_question_retrieval_entry(
    payload: Dict[str, Any] | List[Dict[str, Any]],
    *,
    question_id: Optional[str] = None,
    question_index: Optional[int] = None,
    question_text: Optional[str] = None,
    default_to_first: bool = False,
) -> Optional[Dict[str, Any]]:
    entries = get_per_question_entries(payload)
    if question_id:
        entry = build_retrieval_question_lookup(entries).get(str(question_id))
        if entry is not None:
            return entry
    if question_index is not None:
        if 0 <= question_index < len(entries):
            return entries[question_index]
        return None
    if question_text:
        normalized_text = question_text.strip()
        for entry in entries:
            if str(entry.get("question", "")).strip() == normalized_text:
                return entry
    if default_to_first and entries:
        return entries[0]
    return None


def resolve_representation_text(
    result: Dict[str, Any],
    representation_lookup: Optional[RepresentationLookup] = None,
) -> str:
    if result.get("source_text"):
        return str(result["source_text"])
    if representation_lookup is None:
        return ""

    item_id = str(result.get("item_id", "")).strip()
    item_record = representation_lookup.get("item_id", {}).get(item_id)
    if item_record is not None:
        return str(item_record.get("source_text", ""))

    representation_type = str(result.get("representation_type", "")).strip()
    paper_id = normalize_identifier(result.get("paper_id", ""))
    paper_record = representation_lookup.get("paper", {}).get((representation_type, paper_id))
    if paper_record is None:
        return ""
    return str(paper_record.get("source_text", ""))
