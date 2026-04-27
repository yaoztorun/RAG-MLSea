from __future__ import annotations

from math import log2
from typing import Any

from src.retrieval.data_loading import normalize_iri


def compute_question_metrics(
    candidates: list[dict[str, Any]],
    gold_iri_normalized: str,
    top_k: int = 10,
) -> dict[str, Any]:
    gold = normalize_iri(gold_iri_normalized)
    ranked_ids = [normalize_iri(c.get("normalized_entity_id", c.get("entity_id", ""))) for c in candidates[:top_k]]

    found_gold = gold in ranked_ids
    gold_rank: int | None = None
    for i, eid in enumerate(ranked_ids, start=1):
        if eid == gold:
            gold_rank = i
            break

    def hit_at_k(k: int) -> float:
        return 1.0 if gold in ranked_ids[:k] else 0.0

    mrr = (1.0 / gold_rank) if gold_rank is not None else 0.0
    ndcg = (1.0 / log2(gold_rank + 1)) if gold_rank is not None else 0.0

    return {
        "found_gold": found_gold,
        "gold_rank": gold_rank,
        "hit_at_1": hit_at_k(1),
        "hit_at_5": hit_at_k(5),
        "hit_at_10": hit_at_k(10),
        "mrr": mrr,
        "ndcg": ndcg,
    }


def make_question_result(
    *,
    question_id: str,
    question: str,
    question_type: str,
    difficulty: str | None,
    target_entity_iri: str,
    expected_entity_type: str,
    method_name: str,
    top_k: int,
    candidates: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "difficulty": difficulty,
        "target_entity_iri": target_entity_iri,
        "expected_entity_type": expected_entity_type,
        "method_name": method_name,
        "top_k": top_k,
        "candidates": candidates,
        "found_gold": False,
        "gold_rank": None,
        "hit_at_1": 0.0,
        "hit_at_5": 0.0,
        "hit_at_10": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "warnings": warnings,
    }


def finalize_result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Compute and attach metrics to an existing result dict in-place."""
    iri = normalize_iri(result.get("target_entity_iri", "") or "")
    if not iri:
        result["warnings"] = result.get("warnings", []) + ["no target_entity_iri; metrics set to 0"]
        return result
    metrics = compute_question_metrics(result["candidates"], iri, result.get("top_k", 10))
    result.update(metrics)
    return result


def renumber_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, c in enumerate(candidates, start=1):
        c["new_rank"] = i
    return candidates
