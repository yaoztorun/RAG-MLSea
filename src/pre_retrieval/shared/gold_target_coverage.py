"""
gold_target_coverage.py
Shared utility for checking gold target IRI coverage against canonical records
and ChromaDB collections.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.pre_retrieval.shared.utils import normalize_identifier


_CHROMA_BATCH_SIZE = 5000


def _load_gold_ids(questions_path: Path, entity_type: str) -> Set[str]:
    """Return normalized gold IRIs for a given entity type from the questions file."""
    with questions_path.open("r", encoding="utf-8") as fh:
        questions = json.load(fh)
    gold: Set[str] = set()
    for q in questions:
        if not q.get("is_answerable", True):
            continue
        if q.get("category") != entity_type:
            continue
        iri = q.get("target_entity_iri")
        if iri:
            gold.add(normalize_identifier(str(iri)))
    return gold


def check_gold_target_coverage(
    questions_path: Path,
    master_jsonl_path: Path,
    entity_type: str,
    id_field: str,
) -> Dict[str, Any]:
    """
    Check which gold target IRIs from the question dataset are present in the master JSONL.

    Args:
        questions_path: Path to ml_questions_dataset.json
        master_jsonl_path: Path to the canonical master JSONL (papers/datasets/models)
        entity_type: "paper", "dataset", or "model"
        id_field: "paper_id", "dataset_id", or "model_id"

    Returns:
        dict with keys: requested, found, missing, missing_iris
    """
    gold_ids = _load_gold_ids(questions_path, entity_type)
    label = entity_type.capitalize() + "s"

    found: Set[str] = set()
    with master_jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            nid = normalize_identifier(str(rec.get(id_field, "")))
            if nid in gold_ids:
                found.add(nid)

    missing = gold_ids - found
    print(f"[Gold {label}] Requested: {len(gold_ids)}", flush=True)
    print(f"[Gold {label}] Found in master: {len(found)}", flush=True)
    print(f"[Gold {label}] Missing: {len(missing)}", flush=True)
    if missing:
        sample = sorted(missing)[:5]
        print(f"[Gold {label}] Missing IRIs (first 5): {sample}", flush=True)
    return {
        "requested": len(gold_ids),
        "found": len(found),
        "missing": len(missing),
        "missing_iris": sorted(missing),
    }


def load_gold_ids_for_entity_type(questions_path: Path, entity_type: str) -> Set[str]:
    """Return normalized gold IRIs for the given entity_type. No logging."""
    return _load_gold_ids(questions_path, entity_type)


def check_gold_in_chroma_collection(
    collection: Any,
    gold_ids: Set[str],
    id_field: str,
    entity_type: str,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check how many gold IRIs are present in a ChromaDB collection.
    Uses batched reads to handle large collections (up to 200k vectors).

    Args:
        collection: ChromaDB collection object
        gold_ids: set of normalized gold IRIs to look for
        id_field: metadata field name ("paper_id", "dataset_id", "model_id")
        entity_type: "paper", "dataset", or "model" (for logging)
        collection_name: optional display name for logs

    Returns:
        dict with keys: found, missing, missing_iris
    """
    cname = collection_name or getattr(collection, "name", "?")
    label = entity_type.capitalize() + "s"
    found: Set[str] = set()
    offset = 0

    while True:
        result = collection.get(
            limit=_CHROMA_BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        if not metadatas:
            break
        for meta in metadatas:
            if meta and id_field in meta:
                nid = normalize_identifier(str(meta[id_field]))
                if nid in gold_ids:
                    found.add(nid)
        if len(metadatas) < _CHROMA_BATCH_SIZE:
            break
        offset += _CHROMA_BATCH_SIZE

    missing = gold_ids - found
    print(f"[Gold {label}] In {cname}: {len(found)}/{len(gold_ids)}", flush=True)
    if missing:
        sample = sorted(missing)[:5]
        print(f"[Gold {label}] Missing in {cname} (first 5): {sample}", flush=True)
    return {
        "found": len(found),
        "missing": len(missing),
        "missing_iris": sorted(missing),
    }
