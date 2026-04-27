from __future__ import annotations

from typing import Any

from src.retrieval.config import BEST_REPRESENTATION, DEFAULT_TOP_K
from src.retrieval.data_loading import (
    build_question_index,
    build_raw_candidate,
    entity_type_from_iri,
    load_questions,
    load_top10,
    normalize_iri,
)
from src.retrieval.result_schema import (
    finalize_result_metrics,
    make_question_result,
    renumber_candidates,
)

METHOD_NAME = "pure_semantic_dense"


def _load_all_top10_indexes() -> dict[str, dict[str, Any]]:
    indexes: dict[str, dict[str, Any]] = {}
    for entity_type, representation in BEST_REPRESENTATION.items():
        indexes[entity_type] = load_top10(entity_type, representation)
    return indexes


def _build_result_for_question(
    question: dict[str, Any],
    indexes: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    qid = str(question.get("id", ""))
    raw_iri = question.get("target_entity_iri") or ""
    entity_type = entity_type_from_iri(raw_iri)
    normalized_iri = normalize_iri(raw_iri)
    question_type = question.get("question_type", "") or ""
    difficulty = question.get("difficulty") or None
    is_answerable = question.get("is_answerable", True)
    warnings: list[str] = []

    if not is_answerable or not normalized_iri:
        warnings.append("unanswerable or missing target_entity_iri; no candidates loaded")
        return finalize_result_metrics(
            make_question_result(
                question_id=qid,
                question=question.get("question", ""),
                question_type=question_type,
                difficulty=difficulty,
                target_entity_iri=raw_iri,
                expected_entity_type=entity_type,
                method_name=METHOD_NAME,
                top_k=top_k,
                candidates=[],
                warnings=warnings,
            )
        )

    if entity_type == "unknown":
        warnings.append(f"cannot infer entity type from IRI: {raw_iri}")

    index = indexes.get(entity_type, {})
    representation = BEST_REPRESENTATION.get(entity_type, "")
    entry = index.get(qid)

    if entry is None:
        warnings.append(
            f"question {qid} not found in pre-retrieval top10 for entity_type={entity_type}"
        )
        return finalize_result_metrics(
            make_question_result(
                question_id=qid,
                question=question.get("question", ""),
                question_type=question_type,
                difficulty=difficulty,
                target_entity_iri=raw_iri,
                expected_entity_type=entity_type,
                method_name=METHOD_NAME,
                top_k=top_k,
                candidates=[],
                warnings=warnings,
            )
        )

    candidates = [
        build_raw_candidate(doc, representation)
        for doc in (entry.raw_candidates or [])
    ]
    candidates = renumber_candidates(candidates[:top_k])

    return finalize_result_metrics(
        make_question_result(
            question_id=qid,
            question=entry.question or question.get("question", ""),
            question_type=question_type,
            difficulty=difficulty,
            target_entity_iri=raw_iri,
            expected_entity_type=entity_type,
            method_name=METHOD_NAME,
            top_k=top_k,
            candidates=candidates,
            warnings=warnings,
        )
    )


def run_dense_baseline(
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    if questions is None:
        questions = load_questions()
    indexes = _load_all_top10_indexes()
    return [_build_result_for_question(q, indexes, top_k) for q in questions]
