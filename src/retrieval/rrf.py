from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from src.retrieval.config import (
    DEFAULT_TOP_K,
    RRF_FUSION_GROUPS,
    RRF_K,
)
from src.retrieval.data_loading import (
    build_raw_candidate,
    entity_type_from_iri,
    load_questions,
    load_top10,
    normalize_iri,
)
from src.retrieval.filtering import run_predicate_aware_filtering, run_type_filtering
from src.retrieval.result_schema import (
    finalize_result_metrics,
    make_question_result,
    renumber_candidates,
)


def _load_rrf_indexes() -> dict[str, dict[str, dict[str, Any]]]:
    """Return {entity_type: {representation: {question_id: PreRetrievalEntry}}}."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for entity_type, representations in RRF_FUSION_GROUPS.items():
        result[entity_type] = {}
        for rep in representations:
            index = load_top10(entity_type, rep)
            if index:
                result[entity_type][rep] = index
    return result


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _fuse_question_rrf(
    question_id: str,
    entity_type: str,
    entity_indexes: dict[str, dict[str, Any]],
    top_k: int,
    k: int = RRF_K,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute RRF-fused candidates for a single question."""
    warnings: list[str] = []
    candidate_scores: dict[str, float] = defaultdict(float)
    candidate_original_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    candidate_original_scores: dict[str, dict[str, float]] = defaultdict(dict)
    candidate_meta: dict[str, dict[str, Any]] = {}
    contributing_reps: dict[str, list[str]] = defaultdict(list)

    available_reps = list(entity_indexes.keys())
    if not available_reps:
        warnings.append(f"no fusion representations available for entity_type={entity_type}")
        return [], warnings

    found_in_any = False
    for rep, index in entity_indexes.items():
        entry = index.get(question_id)
        if entry is None:
            warnings.append(f"question {question_id} not found in representation={rep}")
            continue
        found_in_any = True
        for doc in (entry.raw_candidates or []):
            raw_candidate = build_raw_candidate(doc, rep)
            nid = raw_candidate["normalized_entity_id"]
            rank = raw_candidate.get("original_rank")
            # Guard: missing or invalid rank -> treat as just outside the window.
            if rank is None or rank <= 0:
                rank = top_k + 1
            score = raw_candidate.get("original_score")
            candidate_scores[nid] += _rrf_score(rank, k)
            candidate_original_ranks[nid][rep] = rank
            if score is not None:
                candidate_original_scores[nid][rep] = score
            contributing_reps[nid].append(rep)
            if nid not in candidate_meta:
                candidate_meta[nid] = raw_candidate

    if not found_in_any:
        warnings.append(f"question {question_id} not found in any fusion representation")
        return [], warnings

    sorted_ids = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)
    fused: list[dict[str, Any]] = []
    for new_rank, nid in enumerate(sorted_ids[:top_k], start=1):
        base = copy.deepcopy(candidate_meta[nid])
        base["new_rank"] = new_rank
        base["method_score"] = candidate_scores[nid]
        base["rrf_score"] = candidate_scores[nid]
        base["contributing_representations"] = contributing_reps[nid]
        base["original_ranks_by_representation"] = candidate_original_ranks[nid]
        base["original_scores_by_representation"] = candidate_original_scores[nid]
        fused.append(base)

    return fused, warnings


def run_rrf_fusion(
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    if questions is None:
        questions = load_questions()
    rrf_indexes = _load_rrf_indexes()

    output: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question.get("id", ""))
        raw_iri = question.get("target_entity_iri") or ""
        entity_type = entity_type_from_iri(raw_iri)
        question_type = question.get("question_type", "") or ""
        difficulty = question.get("difficulty") or None
        is_answerable = question.get("is_answerable", True)
        warnings: list[str] = []

        if not is_answerable or not normalize_iri(raw_iri):
            warnings.append("unanswerable or missing target_entity_iri; no candidates loaded")
            result = make_question_result(
                question_id=qid,
                question=question.get("question", ""),
                question_type=question_type,
                difficulty=difficulty,
                target_entity_iri=raw_iri,
                expected_entity_type=entity_type,
                method_name="optional_rrf_fusion",
                top_k=top_k,
                candidates=[],
                warnings=warnings,
            )
            output.append(finalize_result_metrics(result))
            continue

        if entity_type == "unknown":
            warnings.append(f"cannot infer entity type from IRI: {raw_iri}; rrf skipped")
            result = make_question_result(
                question_id=qid,
                question=question.get("question", ""),
                question_type=question_type,
                difficulty=difficulty,
                target_entity_iri=raw_iri,
                expected_entity_type=entity_type,
                method_name="optional_rrf_fusion",
                top_k=top_k,
                candidates=[],
                warnings=warnings,
            )
            output.append(finalize_result_metrics(result))
            continue

        entity_indexes = rrf_indexes.get(entity_type, {})
        candidates, fusion_warnings = _fuse_question_rrf(qid, entity_type, entity_indexes, top_k, k)
        warnings.extend(fusion_warnings)

        result = make_question_result(
            question_id=qid,
            question=question.get("question", ""),
            question_type=question_type,
            difficulty=difficulty,
            target_entity_iri=raw_iri,
            expected_entity_type=entity_type,
            method_name="optional_rrf_fusion",
            top_k=top_k,
            candidates=candidates,
            warnings=warnings,
        )
        output.append(finalize_result_metrics(result))

    return output


def run_rrf_symbolic(
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """RRF fusion followed by type filtering then predicate-aware boosting."""
    if questions is None:
        questions = load_questions()

    rrf_results = run_rrf_fusion(questions=questions, top_k=top_k, k=k)

    for r in rrf_results:
        r["method_name"] = "optional_rrf_symbolic"

    type_filtered = run_type_filtering(dense_results=rrf_results, top_k=top_k)

    for r in type_filtered:
        r["method_name"] = "optional_rrf_symbolic"

    predicate_filtered = run_predicate_aware_filtering(dense_results=type_filtered, top_k=top_k)

    for r in predicate_filtered:
        r["method_name"] = "optional_rrf_symbolic"

    return predicate_filtered
