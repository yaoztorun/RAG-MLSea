"""Generate thesis-ready tables and figures from pre-retrieval evaluation results.

Reads from:
  data/retrieval_results/summary.json
  data/retrieval_results/summary_by_difficulty.json

Writes to:
  data/retrieval_results/thesis_tables/
  data/retrieval_results/thesis_figures/

Run:
  python -m src.pre_retrieval.shared.scripts.run_generate_thesis_outputs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.utils import load_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_NAMES: Tuple[str, ...] = ("Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG")
DIFFICULTY_ORDER: Tuple[str, ...] = ("easy", "medium", "hard")
ENTITY_ORDER: Tuple[str, ...] = ("paper", "dataset", "model")
ENTITY_COLORS: Dict[str, str] = {"paper": "#2196F3", "dataset": "#4CAF50", "model": "#FF9800"}
FIGURE_DPI: int = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary_rows(results_dir: Path) -> List[Dict[str, Any]]:
    path = results_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found at {path}")
    payload = load_json(path)
    return list(payload["rows"])


def _load_difficulty_rows(results_dir: Path) -> List[Dict[str, Any]]:
    path = results_dir / "summary_by_difficulty.json"
    if not path.exists():
        raise FileNotFoundError(f"summary_by_difficulty.json not found at {path}")
    payload = load_json(path)

    all_rows: List[Dict[str, Any]] = []
    skipped = 0
    for difficulty in DIFFICULTY_ORDER:
        segment = payload.get("segments", {}).get(difficulty, {})
        for row in segment.get("rows", []):
            assert row.get("evaluated_questions", 0) >= 0, (
                f"Negative evaluated_questions in row: {row}"
            )
            if row.get("evaluated_questions", 0) == 0:
                skipped += 1
                continue
            all_rows.append(row)

    if skipped:
        print(f"[INFO] Skipped {skipped} difficulty rows with evaluated_questions=0 "
              "(dataset/model entities have no easy-difficulty questions by evaluation design).")
    return all_rows


# ---------------------------------------------------------------------------
# Shared computation
# ---------------------------------------------------------------------------

def _compute_best_per_entity(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entity = row["entity_type"]
        if entity not in best or row["NDCG"] > best[entity]["NDCG"]:
            best[entity] = row
    return best


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------

def _md_table(
    headers: List[str],
    rows: List[List[str]],
    best_vals: Optional[Dict[str, float]] = None,
    metric_cols: Optional[List[str]] = None,
) -> str:
    """Build a GFM markdown table. bold_vals maps column name → value to bold."""
    # Build separator row: left-align text cols, right-align metric cols
    metric_set = set(metric_cols or [])
    sep = []
    for h in headers:
        sep.append("---:" if h in metric_set else ":---")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    # Pre-compute which (row_idx, col_idx) cells hold the best value per metric col
    best_set: set = set()
    if best_vals and metric_cols:
        for col_idx, h in enumerate(headers):
            if h in best_vals:
                target = f"{best_vals[h]:.4f}"
                for row_idx, row in enumerate(rows):
                    if col_idx < len(row) and row[col_idx] == target:
                        best_set.add((row_idx, col_idx))

    for row_idx, row in enumerate(rows):
        cells = []
        for col_idx, cell in enumerate(row):
            if (row_idx, col_idx) in best_set:
                cells.append(f"**{cell}**")
            else:
                cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{row[k]:.4f}" if k in METRIC_NAMES and isinstance(row.get(k), float) else row.get(k, ""))
                for k in fieldnames
            })


def _fmt(value: float) -> str:
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_full_comparison_table(rows: List[Dict[str, Any]], tables_dir: Path) -> List[Path]:
    fieldnames = ["entity_type", "representation", *METRIC_NAMES]
    _write_csv(tables_dir / "full_comparison.csv", fieldnames, rows)

    # Compute per-column best values for bolding
    best_vals = {m: max(r[m] for r in rows) for m in METRIC_NAMES}

    headers = ["Entity Type", "Representation", "Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]
    # Map display headers back to original metric names for best_vals lookup
    header_to_metric = {
        "Hit@1": "Hit@1", "Hit@5": "Hit@5", "Hit@10": "Hit@10", "MRR": "MRR", "NDCG": "NDCG",
    }
    best_vals_display = {k: best_vals[v] for k, v in header_to_metric.items()}

    md_rows = [
        [row["entity_type"], row["representation"],
         _fmt(row["Hit@1"]), _fmt(row["Hit@5"]), _fmt(row["Hit@10"]),
         _fmt(row["MRR"]), _fmt(row["NDCG"])]
        for row in rows
    ]
    md = _md_table(headers, md_rows, best_vals=best_vals_display, metric_cols=list(header_to_metric.keys()))
    md_path = tables_dir / "full_comparison.md"
    md_path.write_text(
        "# Table 1: Overall Retrieval Performance\n\n" + md,
        encoding="utf-8",
    )
    return [tables_dir / "full_comparison.csv", md_path]


def build_best_per_entity_table(
    best_per_entity: Dict[str, Dict[str, Any]],
    tables_dir: Path,
) -> List[Path]:
    fieldnames = ["entity_type", "best_representation", *METRIC_NAMES]
    dict_rows = [
        {
            "entity_type": entity,
            "best_representation": best_per_entity[entity]["representation"],
            **{m: best_per_entity[entity][m] for m in METRIC_NAMES},
        }
        for entity in ENTITY_ORDER
        if entity in best_per_entity
    ]
    _write_csv(tables_dir / "best_per_entity.csv", fieldnames, dict_rows)

    headers = ["Entity Type", "Best Representation", "Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]
    md_rows = [
        [r["entity_type"], r["best_representation"],
         _fmt(r["Hit@1"]), _fmt(r["Hit@5"]), _fmt(r["Hit@10"]),
         _fmt(r["MRR"]), _fmt(r["NDCG"])]
        for r in dict_rows
    ]
    md = _md_table(headers, md_rows, metric_cols=["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"])
    md_path = tables_dir / "best_per_entity.md"
    md_path.write_text(
        "# Table 2: Best Representation per Entity Type\n\n" + md,
        encoding="utf-8",
    )
    return [tables_dir / "best_per_entity.csv", md_path]


def build_difficulty_breakdown_table(
    difficulty_rows: List[Dict[str, Any]],
    tables_dir: Path,
) -> List[Path]:
    difficulty_index = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}
    entity_index = {e: i for i, e in enumerate(ENTITY_ORDER)}
    sorted_rows = sorted(
        difficulty_rows,
        key=lambda r: (
            difficulty_index.get(r.get("difficulty", ""), 99),
            entity_index.get(r.get("entity_type", ""), 99),
            r.get("representation", ""),
        ),
    )

    fieldnames = ["difficulty", "entity_type", "representation", "Hit@1", "MRR", "NDCG"]
    _write_csv(tables_dir / "difficulty_breakdown.csv", fieldnames, sorted_rows)

    headers = ["Difficulty", "Entity Type", "Representation", "Hit@1", "MRR", "NDCG"]
    md_rows = [
        [r.get("difficulty", ""), r.get("entity_type", ""), r.get("representation", ""),
         _fmt(r["Hit@1"]), _fmt(r["MRR"]), _fmt(r["NDCG"])]
        for r in sorted_rows
    ]
    md = _md_table(headers, md_rows, metric_cols=["Hit@1", "MRR", "NDCG"])
    md_path = tables_dir / "difficulty_breakdown.md"
    md_path.write_text(
        "# Table 3: Performance by Question Difficulty\n\n" + md,
        encoding="utf-8",
    )
    return [tables_dir / "difficulty_breakdown.csv", md_path]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, stem: Path) -> List[Path]:
    """Save figure as both PNG and PDF at FIGURE_DPI."""
    paths = []
    for ext in (".png", ".pdf"):
        out = stem.with_suffix(ext)
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        paths.append(out)
    return paths


def _simple_bar_chart(
    ax: plt.Axes,
    x_labels: List[str],
    values: List[float],
    title: str,
    y_label: str,
    color: str,
    bar_width: float = 0.5,
) -> None:
    x = np.arange(len(x_labels))
    bars = ax.bar(x, values, bar_width, color=color)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def _grouped_bar_chart(
    ax: plt.Axes,
    x_labels: List[str],
    groups: Dict[str, Dict[str, float]],
    group_order: List[str],
    title: str,
    y_label: str,
    colors: Dict[str, str],
    bar_width: float = 0.25,
) -> None:
    x = np.arange(len(x_labels))
    n = len(group_order)
    offset = (n - 1) * bar_width / 2

    for i, group_name in enumerate(group_order):
        group_data = groups.get(group_name, {})
        heights = [group_data.get(label, 0.0) for label in x_labels]
        positions = x - offset + i * bar_width
        bars = ax.bar(positions, heights, bar_width - 0.02,
                      label=group_name, color=colors.get(group_name))
        for bar, val in zip(bars, heights):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.015,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _entity_label(entity: str) -> str:
    return entity.capitalize()


def build_ndcg_figures(
    rows: List[Dict[str, Any]],
    figures_dir: Path,
    repr_orders: Dict[str, List[str]],
) -> List[Path]:
    written: List[Path] = []
    entity_rows: Dict[str, Dict[str, float]] = {}
    for row in rows:
        entity_rows.setdefault(row["entity_type"], {})[row["representation"]] = row["NDCG"]

    label_map = {"paper": "1A", "dataset": "1B", "model": "1C"}
    for entity in ENTITY_ORDER:
        order = repr_orders.get(entity, [])
        values = [entity_rows.get(entity, {}).get(r, 0.0) for r in order]
        fig, ax = plt.subplots(figsize=(10, 5))
        _simple_bar_chart(
            ax, order, values,
            title=f"Figure {label_map[entity]}: NDCG Comparison \u2014 {_entity_label(entity)} Representations",
            y_label="NDCG",
            color=ENTITY_COLORS[entity],
        )
        plt.tight_layout()
        written.extend(_save_figure(fig, figures_dir / f"ndcg_{entity}"))
        plt.close(fig)
    return written


def build_hit1_figures(
    rows: List[Dict[str, Any]],
    figures_dir: Path,
    repr_orders: Dict[str, List[str]],
) -> List[Path]:
    written: List[Path] = []
    entity_rows: Dict[str, Dict[str, float]] = {}
    for row in rows:
        entity_rows.setdefault(row["entity_type"], {})[row["representation"]] = row["Hit@1"]

    label_map = {"paper": "2A", "dataset": "2B", "model": "2C"}
    for entity in ENTITY_ORDER:
        order = repr_orders.get(entity, [])
        values = [entity_rows.get(entity, {}).get(r, 0.0) for r in order]
        fig, ax = plt.subplots(figsize=(10, 5))
        _simple_bar_chart(
            ax, order, values,
            title=f"Figure {label_map[entity]}: Hit@1 Comparison \u2014 {_entity_label(entity)} Representations",
            y_label="Hit@1",
            color=ENTITY_COLORS[entity],
        )
        plt.tight_layout()
        written.extend(_save_figure(fig, figures_dir / f"hit1_{entity}"))
        plt.close(fig)
    return written


def build_difficulty_figure(
    difficulty_rows: List[Dict[str, Any]],
    best_per_entity: Dict[str, Dict[str, Any]],
    figures_dir: Path,
) -> List[Path]:
    # Build groups: entity_type -> {difficulty -> NDCG} using only best repr per entity
    groups: Dict[str, Dict[str, float]] = {}
    for row in difficulty_rows:
        entity = row.get("entity_type", "")
        if entity not in best_per_entity:
            continue
        best_repr = best_per_entity[entity]["representation"]
        if row.get("representation") != best_repr:
            continue
        groups.setdefault(entity, {})[row["difficulty"]] = row["NDCG"]

    present_entities = [e for e in ENTITY_ORDER if e in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar_chart(
        ax,
        x_labels=list(DIFFICULTY_ORDER),
        groups=groups,
        group_order=present_entities,
        title="Figure 3: NDCG by Question Difficulty (Best Representation per Entity)",
        y_label="NDCG",
        colors=ENTITY_COLORS,
    )
    plt.tight_layout()
    stem = figures_dir / "best_repr_difficulty_breakdown_ndcg"
    written = _save_figure(fig, stem)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def build_readme(
    tables_dir: Path,
    best_per_entity: Dict[str, Dict[str, Any]],
) -> Path:
    best_lines = "\n".join(
        f"- **{entity}**: `{best_per_entity[entity]['representation']}` "
        f"(NDCG = {best_per_entity[entity]['NDCG']:.4f})"
        for entity in ENTITY_ORDER
        if entity in best_per_entity
    )
    content = f"""\
# Thesis Evaluation Outputs — README

## Primary Ranking Metric

**NDCG** (Normalized Discounted Cumulative Gain) is the primary ranking metric.
It measures rank-weighted retrieval relevance and penalizes correct answers that
appear lower in the result list. Higher NDCG indicates better overall ranking quality.

## Input Files

All outputs are derived from exactly two authoritative sources:

| File | Description |
|------|-------------|
| `data/retrieval_results/summary.json` | Overall metrics per entity type and representation (16 rows) |
| `data/retrieval_results/summary_by_difficulty.json` | Per-difficulty metrics; rows with `evaluated_questions=0` are excluded |

No existing result files are modified.

## Best Representation per Entity Type (by NDCG)

{best_lines}

## Tables

### Table 1 — `full_comparison.csv` / `full_comparison.md`
All 16 entity-type × representation combinations with Hit@1, Hit@5, Hit@10, MRR, NDCG.
Best value per metric column is **bolded** in the Markdown version.

### Table 2 — `best_per_entity.csv` / `best_per_entity.md`
One row per entity type showing the representation that achieved the highest NDCG.
Use this for the summary comparison in the thesis.

### Table 3 — `difficulty_breakdown.csv` / `difficulty_breakdown.md`
Per-difficulty (easy / medium / hard) metrics for all representations with evaluated questions.
**Note:** Dataset and model entities have no easy-difficulty questions by evaluation design
(easy questions target paper entities only). Their easy rows are omitted.

## Figures

All figures are saved as both high-resolution PNG (300 dpi) and PDF for direct thesis inclusion.

### Figures 1A–1C — `ndcg_paper`, `ndcg_dataset`, `ndcg_model`
NDCG bar charts, one per entity type, showing performance across that entity's representations.
Representations are kept separate per entity to avoid misleading cross-entity comparisons.

### Figures 2A–2C — `hit1_paper`, `hit1_dataset`, `hit1_model`
Same structure as Figures 1A–1C but for Hit@1.

### Figure 3 — `best_repr_difficulty_breakdown_ndcg`
Grouped bar chart: x-axis = difficulty level (easy / medium / hard),
groups = entity type, y-axis = NDCG.
Only the best representation per entity type is shown.
Missing easy bars for dataset and model are intentional (see Table 3 note above).
"""
    readme_path = tables_dir / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready tables and figures from retrieval evaluation results."
    )
    parser.add_argument("--config", default=None, help="Path to pipeline config JSON.")
    parser.add_argument("--results-dir", default=None, help="Override retrieval results directory.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    eval_cfg = config["evaluation"]
    results_dir = resolve_repo_path(args.results_dir or eval_cfg["output_dir"])

    tables_dir = results_dir / "thesis_tables"
    figures_dir = results_dir / "thesis_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    repr_orders: Dict[str, List[str]] = {
        "paper":   eval_cfg.get("representation_order", []),
        "dataset": eval_cfg.get("dataset_representation_order", []),
        "model":   eval_cfg.get("model_representation_order", []),
    }

    # Load data
    try:
        rows = _load_summary_rows(results_dir)
        difficulty_rows = _load_difficulty_rows(results_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    best_per_entity = _compute_best_per_entity(rows)

    # Log best representations
    for entity in ENTITY_ORDER:
        if entity in best_per_entity:
            br = best_per_entity[entity]
            print(f"[BEST] {entity}: {br['representation']} (NDCG={br['NDCG']:.4f})")

    # Build tables
    written: List[Path] = []
    written.extend(build_full_comparison_table(rows, tables_dir))
    written.extend(build_best_per_entity_table(best_per_entity, tables_dir))
    written.extend(build_difficulty_breakdown_table(difficulty_rows, tables_dir))
    written.append(build_readme(tables_dir, best_per_entity))

    # Build figures
    written.extend(build_ndcg_figures(rows, figures_dir, repr_orders))
    written.extend(build_hit1_figures(rows, figures_dir, repr_orders))
    written.extend(build_difficulty_figure(difficulty_rows, best_per_entity, figures_dir))

    print(f"\nWrote {len(written)} files:")
    for path in written:
        print(f"  {path.relative_to(results_dir.parent.parent)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
