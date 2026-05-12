from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.retrieval.config import METRIC_NAMES
from src.retrieval.result_schema import compute_question_metrics, finalize_result_metrics


def _zero_metrics() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


def _avg_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return _zero_metrics()
    totals = _zero_metrics()
    for v in values:
        for name in METRIC_NAMES:
            totals[name] += v.get(name, 0.0)
    n = len(values)
    return {name: totals[name] / n for name in METRIC_NAMES}


def _question_metric_dict(result: dict[str, Any]) -> dict[str, float]:
    return {
        "Hit@1": result.get("hit_at_1", 0.0),
        "Hit@5": result.get("hit_at_5", 0.0),
        "Hit@10": result.get("hit_at_10", 0.0),
        "MRR": result.get("mrr", 0.0),
        "NDCG": result.get("ndcg", 0.0),
    }


def _is_evaluable(result: dict[str, Any]) -> bool:
    iri = (result.get("target_entity_iri") or "").strip()
    return bool(iri) and iri != "None"


def evaluate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable = [r for r in results if _is_evaluable(r)]
    not_evaluable_count = len(results) - len(evaluable)

    overall_metrics_list: list[dict[str, float]] = []
    by_difficulty: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_entity_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_question_type: dict[str, list[dict[str, float]]] = defaultdict(list)

    for result in evaluable:
        m = _question_metric_dict(result)
        overall_metrics_list.append(m)

        difficulty = result.get("difficulty") or "unknown"
        by_difficulty[difficulty].append(m)

        etype = result.get("expected_entity_type") or "unknown"
        by_entity_type[etype].append(m)

        qtype = result.get("question_type") or "unknown"
        by_question_type[qtype].append(m)

    def _segment_summary(seg: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
        return {
            key: {
                "count": len(vals),
                **_avg_metrics(vals),
            }
            for key, vals in sorted(seg.items())
        }

    return {
        "total_questions": len(results),
        "evaluable_questions": len(evaluable),
        "skipped_questions": not_evaluable_count,
        "overall": {
            "count": len(evaluable),
            **_avg_metrics(overall_metrics_list),
        },
        "by_difficulty": _segment_summary(by_difficulty),
        "by_entity_type": _segment_summary(by_entity_type),
        "by_question_type": _segment_summary(by_question_type),
    }


def ensure_metrics_on_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-run finalize_result_metrics on results that may lack metrics."""
    return [finalize_result_metrics(r) for r in results]
