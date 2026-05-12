from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.retrieval.config import (
    ALL_METHODS,
    INTERPRETATION_LABELS,
    METHOD_GROUPS,
    METRIC_NAMES,
    RETRIEVAL_OUTPUT_DIR,
)


def _load_metrics(method_dir: Path) -> dict[str, Any] | None:
    p = method_dir / "metrics.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _flat_row(method_name: str, metrics_payload: dict[str, Any]) -> dict[str, Any]:
    overall = metrics_payload.get("overall", {})
    row: dict[str, Any] = {"method": method_name}
    row["evaluated_questions"] = overall.get("count", 0)
    for name in METRIC_NAMES:
        row[name] = round(overall.get(name, 0.0), 4)
    return row


def build_summary_json(output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in ALL_METHODS:
        method_dir = output_dir / method
        payload = _load_metrics(method_dir)
        if payload is None:
            continue
        summary[method] = payload
    return summary


def build_summary_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        method_dir = output_dir / method
        payload = _load_metrics(method_dir)
        if payload is None:
            continue
        rows.append(_flat_row(method, payload))
    return rows


def _by_dimension(summary: dict[str, Any], dimension: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, payload in summary.items():
        result[method] = payload.get(dimension, {})
    return result


def build_delta_vs_dense(summary: dict[str, Any]) -> dict[str, Any]:
    """Per-metric delta of each method vs pure_semantic_dense."""
    baseline = summary.get("pure_semantic_dense", {}).get("overall", {})
    result: dict[str, Any] = {}
    for method, payload in summary.items():
        overall = payload.get("overall", {})
        result[method] = {
            f"delta_{n}": round(overall.get(n, 0.0) - baseline.get(n, 0.0), 4)
            for n in METRIC_NAMES
        }
    return result


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Retrieval Stage — Method Comparison\n\n")
        header = " | ".join(fieldnames)
        fh.write(f"| {header} |\n")
        fh.write("|" + "|".join("---" for _ in fieldnames) + "|\n")
        for row in rows:
            vals = " | ".join(str(row.get(f, "")) for f in fieldnames)
            fh.write(f"| {vals} |\n")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------
# Thesis table builders
# -----------------------------------------------------------------------

def _thesis_main_comparison_rows(
    summary: dict[str, Any],
    delta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        if method not in summary:
            continue
        overall = summary[method].get("overall", {})
        d = delta.get(method, {})
        rows.append({
            "method": method,
            "method_group": METHOD_GROUPS.get(method, ""),
            "Hit@1": round(overall.get("Hit@1", 0.0), 4),
            "Hit@5": round(overall.get("Hit@5", 0.0), 4),
            "Hit@10": round(overall.get("Hit@10", 0.0), 4),
            "MRR": round(overall.get("MRR", 0.0), 4),
            "NDCG": round(overall.get("NDCG", 0.0), 4),
            "delta_NDCG_vs_dense": d.get("delta_NDCG", 0.0),
            "interpretation_label": INTERPRETATION_LABELS.get(method, ""),
        })
    return rows


def _thesis_by_difficulty_ndcg_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        if method not in summary:
            continue
        by_diff = summary[method].get("by_difficulty", {})
        rows.append({
            "method": method,
            "easy_NDCG": round(by_diff.get("easy", {}).get("NDCG", 0.0), 4),
            "medium_NDCG": round(by_diff.get("medium", {}).get("NDCG", 0.0), 4),
            "hard_NDCG": round(by_diff.get("hard", {}).get("NDCG", 0.0), 4),
            "unknown_NDCG": round(by_diff.get("unknown", {}).get("NDCG", 0.0), 4),
        })
    return rows


def _thesis_precision_recall_tradeoff_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        if method not in summary:
            continue
        overall = summary[method].get("overall", {})
        h1 = round(overall.get("Hit@1", 0.0), 4)
        h10 = round(overall.get("Hit@10", 0.0), 4)
        rows.append({
            "method": method,
            "Hit@1": h1,
            "Hit@10": h10,
            "gap_Hit10_minus_Hit1": round(h10 - h1, 4),
            "interpretation": INTERPRETATION_LABELS.get(method, ""),
        })
    return rows


def _thesis_by_entity_type_ndcg_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        if method not in summary:
            continue
        by_et = summary[method].get("by_entity_type", {})
        rows.append({
            "method": method,
            "paper_NDCG": round(by_et.get("paper", {}).get("NDCG", 0.0), 4),
            "dataset_NDCG": round(by_et.get("dataset", {}).get("NDCG", 0.0), 4),
            "model_NDCG": round(by_et.get("model", {}).get("NDCG", 0.0), 4),
        })
    return rows


def _thesis_by_question_type_ndcg_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        if method not in summary:
            continue
        by_qt = summary[method].get("by_question_type", {})
        for qtype, stats in sorted(by_qt.items()):
            rows.append({
                "method": method,
                "question_type": qtype,
                "count": stats.get("count", 0),
                "NDCG": round(stats.get("NDCG", 0.0), 4),
                "Hit@1": round(stats.get("Hit@1", 0.0), 4),
                "Hit@10": round(stats.get("Hit@10", 0.0), 4),
            })
    return rows


def _write_interpretation_md(
    summary: dict[str, Any],
    delta: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = [
        "# Retrieval Stage — Thesis Interpretation",
        "",
        "## Method Overview",
        "",
        "| Method | Group | NDCG | delta_Hit@1 | delta_NDCG | Interpretation |",
        "|---|---|---|---|---|---|",
    ]
    for method in ALL_METHODS:
        if method not in summary:
            continue
        overall = summary[method].get("overall", {})
        d = delta.get(method, {})
        ndcg = round(overall.get("NDCG", 0.0), 4)
        d_h1 = d.get("delta_Hit@1", 0.0)
        d_ndcg = d.get("delta_NDCG", 0.0)
        group = METHOD_GROUPS.get(method, "")
        label = INTERPRETATION_LABELS.get(method, "")
        sign_h1 = "+" if d_h1 >= 0 else ""
        sign_ndcg = "+" if d_ndcg >= 0 else ""
        lines.append(
            f"| {method} | {group} | {ndcg} | {sign_h1}{d_h1} | {sign_ndcg}{d_ndcg} | {label} |"
        )

    lines += [
        "",
        "## Framing",
        "",
        "The retrieval stage compares three strategy families:",
        "",
        "**1. Pure Semantic (Dense)**",
        "",
        "- `pure_semantic_dense` is the reference baseline.",
        "- Uses the best pre-retrieval representation per entity type:",
        "  `enriched_metadata` for papers, `dataset_title_only` for datasets,",
        "  `model_predicate_filtered` for models.",
        "- Strong starting point because pre-retrieval tuned the representations.",
        "",
        "**2. Semantic + Symbolic Hybrid**",
        "",
        "- `hybrid_type_filtering`: applies expected entity-type constraint after dense retrieval.",
        "  Because pre-retrieval outputs are already entity-type-specific, this acts as a",
        "  control. If metrics match the dense baseline, it confirms the collection is pure",
        "  and type filtering adds no information.",
        "- `hybrid_type_onehop_filtering`: applies type filter, then boosts candidates with",
        "  richer graph connections (non-empty tasks/datasets/methods/implementations or",
        "  Linked Entities in source text). Tests whether better graph-connected entities",
        "  are more relevant regardless of question type.",
        "- `hybrid_predicate_aware_filtering`: boosts candidates based on question-type-specific",
        "  predicate signals. Tests whether question intent can guide reranking via symbolic",
        "  predicates (e.g., boost candidates with `tasks` for task-related questions).",
        "",
        "**3. Optional RRF Fusion (Exploratory)**",
        "",
        "- `optional_rrf_fusion`: fuses multiple representation rankings via Reciprocal Rank",
        "  Fusion (k=60). Trades Hit@1 for broader recall at Hit@10. Not the main strategy.",
        "- `optional_rrf_symbolic`: RRF followed by type filter and predicate-aware boosting.",
        "  Tests whether symbolic post-processing recovers precision lost by fusion.",
        "",
        "## What to Look For",
        "",
        "- **hybrid_type_filtering == pure_semantic_dense**: expected, because pre-retrieval",
        "  outputs are entity-type-pure. Use as a sanity check, not a performance claim.",
        "- **Hit@1 vs Hit@10 gap**: large gap means the answer exists in the candidate pool",
        "  but not at rank 1. Post-retrieval re-ranking can close this gap.",
        "- **Hard question NDCG**: key diagnostic. If hybrid methods do not improve hard",
        "  questions, the bottleneck is representation quality, not ranking strategy.",
        "- **Dataset entity type**: consistently lower NDCG than papers/models.",
        "  Structural: dataset metadata is sparse in MLSea.",
        "",
        "## Important Cautions",
        "",
        "- Do not overclaim hybrid improvements. The dense baseline is already strong because",
        "  pre-retrieval selected the best representations.",
        "- Predicate-aware boosting can hurt some question types by reordering away from the",
        "  gold answer when predicate evidence is sparse or ambiguous.",
        "- RRF is included as exploratory fusion to test recall broadening; it is not a main",
        "  thesis claim and should be presented accordingly.",
        "- Re-ranking is NOT part of this stage. Post-retrieval re-ranking operates on the",
        "  fixed top-K candidates produced here.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_thesis_tables(summary: dict[str, Any], output_dir: Path) -> None:
    tables_dir = output_dir / "thesis_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    delta = build_delta_vs_dense(summary)

    write_summary_csv(
        _thesis_main_comparison_rows(summary, delta),
        tables_dir / "retrieval_main_comparison.csv",
    )
    write_summary_csv(
        _thesis_by_difficulty_ndcg_rows(summary),
        tables_dir / "retrieval_by_difficulty_ndcg.csv",
    )
    write_summary_csv(
        _thesis_precision_recall_tradeoff_rows(summary),
        tables_dir / "retrieval_precision_recall_tradeoff.csv",
    )
    write_summary_csv(
        _thesis_by_entity_type_ndcg_rows(summary),
        tables_dir / "retrieval_by_entity_type_ndcg.csv",
    )
    write_summary_csv(
        _thesis_by_question_type_ndcg_rows(summary),
        tables_dir / "retrieval_by_question_type_ndcg.csv",
    )
    _write_interpretation_md(summary, delta, tables_dir / "retrieval_interpretation.md")


def aggregate_all_methods(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = RETRIEVAL_OUTPUT_DIR

    summary = build_summary_json(output_dir)
    if not summary:
        return

    rows = build_summary_rows(output_dir)

    write_json(summary, output_dir / "summary.json")
    write_summary_csv(rows, output_dir / "summary.csv")
    write_summary_md(rows, output_dir / "summary.md")
    write_json(_by_dimension(summary, "by_difficulty"), output_dir / "summary_by_difficulty.json")
    write_json(_by_dimension(summary, "by_entity_type"), output_dir / "summary_by_entity_type.json")
    write_json(_by_dimension(summary, "by_question_type"), output_dir / "summary_by_question_type.json")

    delta = build_delta_vs_dense(summary)
    write_json(delta, output_dir / "summary_delta_vs_dense.json")
    delta_rows = [{"method": m, **v} for m, v in delta.items()]
    write_summary_csv(delta_rows, output_dir / "summary_delta_vs_dense.csv")

    write_thesis_tables(summary, output_dir)
