from __future__ import annotations

import copy
from typing import Any

from src.retrieval.config import DEFAULT_TOP_K
from src.retrieval.data_loading import entity_type_from_iri, load_questions
from src.retrieval.dense_baseline import run_dense_baseline
from src.retrieval.result_schema import finalize_result_metrics, renumber_candidates

_TASK_TYPES = {
    "paper_to_tasks",
    "paper_by_task_pair",
    "paper_to_task_count",
    "paper_by_author_and_task",
    "dataset_to_tasks",
    "task_to_dataset",
    "tasks_to_dataset",
    "semantic_task_to_dataset",
    "dataset_to_task_count",
    "dataset_to_task_membership",
}
_IMPLEMENTATION_TYPES = {"paper_to_implementation"}
_YEAR_TYPES = {"paper_to_publication_year", "dataset_to_publication_year"}
_REPOSITORY_TYPES = {"repository_to_model", "semantic_repository_to_model"}
_FAMILY_TYPES = {"model_family_variant", "comparative_model_variant"}
_KEYWORD_TYPES = {"paper_to_keywords"}

# One-hop graph-connectivity metadata fields used by hybrid_type_onehop_filtering.
_ONEHOP_FIELDS = ("tasks", "datasets", "methods", "metrics", "implementations")


def _boost_by_predicate(
    candidates: list[dict[str, Any]],
    question_type: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Split candidates into boosted/non-boosted groups based on question_type.

    Returns (reordered_candidates, evidence_found).
    """
    def _meta(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("metadata") or {}

    def _nonempty(val: Any) -> bool:
        return bool(val)

    if question_type in _TASK_TYPES:
        boosted = [c for c in candidates if _nonempty(_meta(c).get("tasks"))]
        rest = [c for c in candidates if not _nonempty(_meta(c).get("tasks"))]
    elif question_type in _IMPLEMENTATION_TYPES:
        boosted = [c for c in candidates if _nonempty(_meta(c).get("implementations"))]
        rest = [c for c in candidates if not _nonempty(_meta(c).get("implementations"))]
    elif question_type in _YEAR_TYPES:
        boosted = [c for c in candidates if _meta(c).get("publication_year") is not None]
        rest = [c for c in candidates if _meta(c).get("publication_year") is None]
    elif question_type in _KEYWORD_TYPES:
        boosted = [c for c in candidates if _nonempty(_meta(c).get("keywords"))]
        rest = [c for c in candidates if not _nonempty(_meta(c).get("keywords"))]
    elif question_type in _REPOSITORY_TYPES or question_type in _FAMILY_TYPES:
        source_text = [c.get("metadata", {}).get("source_text", "") for c in candidates]
        boosted = [c for c, st in zip(candidates, source_text) if st and "Linked Entities" in st]
        rest = [c for c, st in zip(candidates, source_text) if not (st and "Linked Entities" in st)]
    else:
        return candidates, False

    evidence_found = bool(boosted)
    return boosted + rest, evidence_found


def _onehop_richness(candidate: dict[str, Any]) -> int:
    """Count non-empty one-hop / linked-entity metadata fields."""
    meta = candidate.get("metadata") or {}
    score = sum(1 for f in _ONEHOP_FIELDS if meta.get(f))
    if meta.get("source_text") and "Linked Entities" in meta["source_text"]:
        score += 2
    return score


def run_type_filtering(
    dense_results: list[dict[str, Any]] | None = None,
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    if dense_results is None:
        dense_results = run_dense_baseline(questions=questions, top_k=top_k)

    output: list[dict[str, Any]] = []
    for result in dense_results:
        r = copy.deepcopy(result)
        r["method_name"] = "hybrid_type_filtering"
        warnings = list(r.get("warnings", []))

        expected_type = r.get("expected_entity_type", "")
        if not expected_type or expected_type == "unknown":
            iri = r.get("target_entity_iri", "")
            expected_type = entity_type_from_iri(iri)

        if expected_type == "unknown":
            warnings.append("cannot infer expected entity type; type filtering skipped")
            r["warnings"] = warnings
            output.append(finalize_result_metrics(r))
            continue

        original = r.get("candidates", [])
        filtered = [c for c in original if c.get("entity_type") == expected_type]

        if not filtered:
            warnings.append(
                f"no candidates matched entity_type={expected_type}; "
                "reverting to unfiltered candidates"
            )
            filtered = original

        filtered = renumber_candidates(filtered[:top_k])
        r["candidates"] = filtered
        r["warnings"] = warnings
        output.append(finalize_result_metrics(r))

    return output


def run_hybrid_type_onehop_filtering(
    dense_results: list[dict[str, Any]] | None = None,
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Type filter then general graph-connectivity boosting.

    Hypothesis: among type-correct candidates, those with richer one-hop
    graph connections (non-empty tasks / datasets / methods / metrics /
    implementations, or "Linked Entities" in source text) are more likely
    to be the answer.  The boost is question-type-agnostic so it is
    genuinely distinct from hybrid_predicate_aware_filtering.
    """
    if dense_results is None:
        dense_results = run_dense_baseline(questions=questions, top_k=top_k)

    output: list[dict[str, Any]] = []
    for result in dense_results:
        r = copy.deepcopy(result)
        r["method_name"] = "hybrid_type_onehop_filtering"
        warnings = list(r.get("warnings", []))

        expected_type = r.get("expected_entity_type", "")
        if not expected_type or expected_type == "unknown":
            iri = r.get("target_entity_iri", "")
            expected_type = entity_type_from_iri(iri)

        original = r.get("candidates", [])
        if not original:
            r["warnings"] = warnings
            output.append(finalize_result_metrics(r))
            continue

        # Step 1: type filter
        if expected_type != "unknown":
            type_filtered = [c for c in original if c.get("entity_type") == expected_type]
            if not type_filtered:
                warnings.append(
                    f"no candidates matched entity_type={expected_type}; "
                    "reverting to unfiltered for one-hop stage"
                )
                type_filtered = original
        else:
            warnings.append("cannot infer expected entity type; type filter skipped")
            type_filtered = original

        # Step 2: one-hop richness soft boost.
        # Blend semantic score with a small richness bonus so graph connectivity
        # can only shift a candidate when semantic scores are close.
        # epsilon=0.01 means richness can contribute at most 0.07 (7 fields * 0.01)
        # — enough to break near-ties but not enough to override a strong semantic rank.
        scored = [(c, _onehop_richness(c)) for c in type_filtered]
        has_onehop = any(s > 0 for _, s in scored)

        if not has_onehop:
            warnings.append("missing one-hop metadata; preserved dense order")
            reordered = type_filtered
        else:
            epsilon = 0.01

            def _blended(pair: tuple) -> float:
                c, richness = pair
                sem = c.get("original_score")
                if sem is None:
                    # Fall back to inverted rank so sort direction is consistent
                    sem = 1.0 / (c.get("original_rank") or 99)
                return sem + epsilon * richness

            reordered = [c for c, _ in sorted(scored, key=_blended, reverse=True)]

        reordered = renumber_candidates(reordered[:top_k])
        r["candidates"] = reordered
        r["warnings"] = warnings
        output.append(finalize_result_metrics(r))

    return output


def run_predicate_aware_filtering(
    dense_results: list[dict[str, Any]] | None = None,
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    if dense_results is None:
        dense_results = run_dense_baseline(questions=questions, top_k=top_k)

    output: list[dict[str, Any]] = []
    for result in dense_results:
        r = copy.deepcopy(result)
        r["method_name"] = "hybrid_predicate_aware_filtering"
        warnings = list(r.get("warnings", []))
        question_type = r.get("question_type", "") or ""

        original = r.get("candidates", [])
        if not original:
            r["warnings"] = warnings
            output.append(finalize_result_metrics(r))
            continue

        reordered, evidence_found = _boost_by_predicate(original, question_type)

        if not evidence_found:
            warnings.append(
                f"no predicate evidence found for question_type={question_type!r}; "
                "original candidate order preserved"
            )

        reordered = renumber_candidates(reordered[:top_k])
        r["candidates"] = reordered
        r["warnings"] = warnings
        output.append(finalize_result_metrics(r))

    return output
