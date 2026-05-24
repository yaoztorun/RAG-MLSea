from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.retrieval.config import DEFAULT_TOP_K, RETRIEVAL_OUTPUT_DIR
from src.retrieval.data_loading import entity_type_from_iri, load_questions
from src.retrieval.dense_baseline import run_dense_baseline
from src.retrieval.result_schema import finalize_result_metrics, renumber_candidates

# -- Question-type sets aligned with actual dataset question_type values --
_TASK_TYPES = {"paper_task_to_paper", "dataset_task_to_dataset"}
_AUTHOR_TYPES = {"paper_author_to_paper"}
_KEYWORD_TYPES = {"dataset_keyword_to_dataset"}
_REPOSITORY_TYPES = {"model_repository_to_model"}
_FAMILY_TYPES = {"model_family_variant"}
_MODEL_PAPER_TYPES = {"model_paper_to_model"}
_MULTI_METADATA_TYPES = {"paper_multi_metadata", "dataset_multi_metadata", "model_multi_metadata"}

_ALL_PREDICATE_TYPES = (
    _TASK_TYPES | _AUTHOR_TYPES | _KEYWORD_TYPES
    | _REPOSITORY_TYPES | _FAMILY_TYPES | _MODEL_PAPER_TYPES
    | _MULTI_METADATA_TYPES
)

# One-hop graph-connectivity metadata fields used by hybrid_type_onehop_filtering.
_ONEHOP_FIELDS = ("tasks", "datasets", "methods", "metrics", "implementations")

# Conservative epsilon: adds at most 0.02 per unit of lexical overlap.
# Enough to break near-ties in semantic scores; not enough to override large gaps.
_PREDICATE_EPSILON = 0.02

# Generic question words that are not entity-specific identifiers.
_STOPWORDS = frozenset({
    "which", "what", "the", "is", "in", "of", "a", "an", "and", "or",
    "to", "from", "that", "this", "model", "paper", "dataset", "connected",
    "associated", "linked", "related", "uses", "used", "has", "have", "its",
    "with", "for", "by", "on", "at", "be", "are", "was", "were", "not",
    "no", "name", "named", "called", "identified", "known",
})


def _field_overlap(question_text: str, candidate: dict[str, Any], fields: tuple[str, ...]) -> float:
    """Fraction of non-stopword question tokens found in concatenated metadata field values."""
    if not question_text:
        return 0.0
    meta = candidate.get("metadata") or {}
    parts: list[str] = []
    for f in fields:
        v = meta.get(f)
        if not v:
            continue
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        else:
            parts.append(str(v))
    field_str = " ".join(parts).lower()
    if not field_str:
        return 0.0
    tokens = {t.strip("()[].,?'\"/:-") for t in question_text.lower().split()} - _STOPWORDS
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in field_str) / len(tokens)


def _boost_by_predicate(
    candidates: list[dict[str, Any]],
    question_type: str,
    question_text: str = "",
) -> tuple[list[dict[str, Any]], bool]:
    """Soft-rerank candidates by lexical overlap between question and relevant metadata fields.

    Returns (reordered_candidates, evidence_found).
    evidence_found is True if at least one candidate had non-zero overlap.
    """
    if question_type in _TASK_TYPES:
        fields: tuple[str, ...] = ("tasks",)
    elif question_type in _AUTHOR_TYPES:
        fields = ("authors",)
    elif question_type in _KEYWORD_TYPES:
        fields = ("keywords",)
    elif question_type in _REPOSITORY_TYPES:
        fields = ("implementations", "source_text")
    elif question_type in _FAMILY_TYPES:
        fields = ("implementations", "source_text")
    elif question_type in _MODEL_PAPER_TYPES:
        fields = ("source_text",)
    elif question_type in _MULTI_METADATA_TYPES:
        fields = (
            "tasks", "keywords", "authors", "datasets",
            "methods", "metrics", "implementations", "source_text",
        )
    else:
        return candidates, False

    overlaps = [_field_overlap(question_text, c, fields) for c in candidates]
    evidence_found = any(o > 0 for o in overlaps)

    def _blended(idx_c: tuple[int, dict[str, Any]]) -> float:
        idx, c = idx_c
        sem = c.get("original_score") or (1.0 / (c.get("original_rank") or 99))
        return sem + _PREDICATE_EPSILON * overlaps[idx]

    reordered = [c for _, c in sorted(enumerate(candidates), key=_blended, reverse=True)]
    return reordered, evidence_found


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

    # Determine debug output directory based on the calling context.
    # When called from run_rrf_symbolic, input results carry method_name="optional_rrf_symbolic"
    # so we write stats there instead of overwriting the hybrid_predicate stats file.
    source_method = (dense_results[0].get("method_name", "") or "") if dense_results else ""
    if "rrf_symbolic" in source_method:
        _debug_dir_name = "optional_rrf_symbolic"
    else:
        _debug_dir_name = "hybrid_predicate_aware_filtering"

    stats: dict[str, Any] = {
        "epsilon": _PREDICATE_EPSILON,
        "source_method": source_method,
        "total_questions": 0,
        "rule_matched": 0,
        "evidence_found": 0,
        "order_changed": 0,
        "gold_rank_changed": 0,
        "by_question_type": {},
        "changed_ranking_examples": [],
    }

    output: list[dict[str, Any]] = []
    for result in dense_results:
        r = copy.deepcopy(result)
        r["method_name"] = "hybrid_predicate_aware_filtering"
        warnings = list(r.get("warnings", []))
        question_type = r.get("question_type", "") or ""
        question_text = r.get("question", "") or ""

        original = r.get("candidates", [])
        if not original:
            r["warnings"] = warnings
            output.append(finalize_result_metrics(r))
            continue

        stats["total_questions"] += 1
        qt_stats = stats["by_question_type"].setdefault(question_type, {
            "total": 0, "rule_matched": 0, "evidence_found": 0,
            "order_changed": 0, "gold_rank_changed": 0,
        })
        qt_stats["total"] += 1

        rule_matched = question_type in _ALL_PREDICATE_TYPES
        if rule_matched:
            stats["rule_matched"] += 1
            qt_stats["rule_matched"] += 1

        reordered, evidence_found = _boost_by_predicate(original, question_type, question_text)

        if evidence_found:
            stats["evidence_found"] += 1
            qt_stats["evidence_found"] += 1

        if not evidence_found:
            warnings.append(
                f"no predicate evidence found for question_type={question_type!r}; "
                "original candidate order preserved"
            )

        old_ids = [c.get("normalized_entity_id", "") for c in original[:top_k]]
        new_ids = [c.get("normalized_entity_id", "") for c in reordered[:top_k]]
        order_changed = old_ids != new_ids
        if order_changed:
            stats["order_changed"] += 1
            qt_stats["order_changed"] += 1

        old_gold_rank = r.get("gold_rank")

        reordered = renumber_candidates(reordered[:top_k])
        r["candidates"] = reordered
        r["warnings"] = warnings
        r = finalize_result_metrics(r)

        new_gold_rank = r.get("gold_rank")
        gold_rank_changed = old_gold_rank != new_gold_rank
        if gold_rank_changed:
            stats["gold_rank_changed"] += 1
            qt_stats["gold_rank_changed"] += 1
            if len(stats["changed_ranking_examples"]) < 5:
                stats["changed_ranking_examples"].append({
                    "question_id": r.get("question_id", ""),
                    "question_type": question_type,
                    "question": question_text[:120],
                    "old_gold_rank": old_gold_rank,
                    "new_gold_rank": new_gold_rank,
                })

        output.append(r)

    # Write debug stats
    debug_dir = Path(RETRIEVAL_OUTPUT_DIR) / _debug_dir_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "debug_stats.json"
    with debug_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"[predicate] debug stats -> {debug_path}")
    print(
        f"  total={stats['total_questions']}  rule_matched={stats['rule_matched']}  "
        f"evidence_found={stats['evidence_found']}  order_changed={stats['order_changed']}  "
        f"gold_rank_changed={stats['gold_rank_changed']}"
    )

    return output
