"""
Build pre-retrieval result artifacts.

Reads aggregated outputs from data/results/pre_retrieval/ and writes:
  - summary_all_representations.csv
  - summary_all_representations.json
  - thesis_tables/pre_retrieval_representation_metrics.tex
  - figures/overall_{metric}_by_representation.pdf      (3 figures)
  - figures/difficulty_{metric}_by_representation.pdf   (3 figures)
  - figures/category_{metric}_by_representation.pdf     (3 figures)

Does NOT run retrieval, embeddings, or ChromaDB.

Usage:
    python scripts/build_pre_retrieval_result_artifacts.py
    python scripts/build_pre_retrieval_result_artifacts.py \\
        --input-dir  data/results/pre_retrieval \\
        --output-dir data/results/pre_retrieval

Requirements: matplotlib, numpy, pandas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive PDF backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]
FIGURE_METRICS = ["NDCG", "Hit@10", "MRR"]

ENTITY_TYPES = ["paper", "dataset", "model"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
CATEGORY_ORDER = ["paper", "dataset", "model", "implementation", "multihop", "semantic"]

PAPER_REPR_ORDER = [
    "title_only", "abstract_only", "title_abstract",
    "predicate_filtered", "enriched_metadata", "one_hop",
]
DATASET_REPR_ORDER = [
    "dataset_title_only", "dataset_metadata",
    "dataset_predicate_filtered", "dataset_enriched_metadata",
]
MODEL_REPR_ORDER = [
    "model_title_only", "model_metadata",
    "model_predicate_filtered", "model_enriched_metadata",
]
REPR_ORDER = PAPER_REPR_ORDER + DATASET_REPR_ORDER + MODEL_REPR_ORDER
ENTITY_REPR_ORDER = {
    "paper": PAPER_REPR_ORDER,
    "dataset": DATASET_REPR_ORDER,
    "model": MODEL_REPR_ORDER,
}

SHORT_LABELS: dict[str, str] = {
    "title_only": "title",
    "abstract_only": "abstract",
    "title_abstract": "title+abs",
    "predicate_filtered": "pred_filt",
    "enriched_metadata": "enr_meta",
    "one_hop": "1-hop",
    "dataset_title_only": "d_title",
    "dataset_metadata": "d_meta",
    "dataset_predicate_filtered": "d_pred",
    "dataset_enriched_metadata": "d_enr",
    "model_title_only": "m_title",
    "model_metadata": "m_meta",
    "model_predicate_filtered": "m_pred",
    "model_enriched_metadata": "m_enr",
}

METRIC_FILE_NAMES: dict[str, str] = {
    "NDCG": "ndcg",
    "Hit@10": "hit_at_10",
    "MRR": "mrr",
}

ENTITY_COLORS = {
    "paper": "#4C72B0",
    "dataset": "#DD8452",
    "model": "#55A868",
}
DIFFICULTY_COLORS = {
    "easy": "#4C72B0",
    "medium": "#DD8452",
    "hard": "#C44E52",
}
CATEGORY_COLORS = {
    "paper": "#4C72B0",
    "dataset": "#DD8452",
    "model": "#55A868",
    "implementation": "#8172B2",
    "multihop": "#937860",
    "semantic": "#C44E52",
}

RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "figure.dpi": 150,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(repr_name: str) -> str:
    return SHORT_LABELS.get(repr_name, repr_name)


def _repr_order_key(r: str) -> int:
    try:
        return REPR_ORDER.index(r)
    except ValueError:
        return len(REPR_ORDER)


def _entity_order_key(e: str) -> int:
    try:
        return ENTITY_TYPES.index(e)
    except ValueError:
        return len(ENTITY_TYPES)


def _load_json_safe(path: Path, label: str) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {label} not found at {path}", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(input_dir: Path) -> pd.DataFrame:
    """Load overall metrics per representation. Falls back to individual results.json."""
    data = _load_json_safe(input_dir / "summary.json", "summary.json")
    if data and data.get("rows"):
        df = pd.DataFrame(data["rows"])
    else:
        df = _load_summary_from_results(input_dir)

    if df.empty:
        return df

    df["_et_order"] = df.get("entity_type", pd.Series(dtype=str)).map(_entity_order_key).fillna(99)
    df["_repr_order"] = df["representation"].map(_repr_order_key).fillna(99)
    df = df.sort_values(["_et_order", "_repr_order"]).drop(columns=["_et_order", "_repr_order"])
    return df.reset_index(drop=True)


def _load_summary_from_results(input_dir: Path) -> pd.DataFrame:
    rows = []
    for entity_type in ENTITY_TYPES:
        entity_dir = input_dir / f"{entity_type}_results"
        if not entity_dir.is_dir():
            continue
        for results_path in sorted(entity_dir.glob("*/results.json")):
            try:
                d = json.loads(results_path.read_text(encoding="utf-8"))
                metrics = d.get("metrics", {})
                rows.append({
                    "entity_type": entity_type,
                    "representation": d.get("representation_type", results_path.parent.name),
                    **{m: float(metrics.get(m, 0.0)) for m in METRICS},
                })
            except Exception as exc:
                print(f"  WARNING: could not read {results_path}: {exc}", file=sys.stderr)
    if not rows:
        print("  WARNING: no results.json files found in input_dir", file=sys.stderr)
    return pd.DataFrame(rows)


def load_by_segment(input_dir: Path, filename: str) -> pd.DataFrame:
    """Load difficulty or category breakdown. Returns flat DataFrame."""
    data = _load_json_safe(input_dir / filename, filename)
    if data is None:
        return pd.DataFrame()
    rows: list[dict] = []
    for seg in data.get("segments", {}).values():
        rows.extend(seg.get("rows", []))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _filter_entity(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if "entity_type" not in df.columns:
        return df
    return df[df["entity_type"] == entity_type].copy()


def _ordered_reprs(df: pd.DataFrame, entity_type: str) -> list[str]:
    canonical = ENTITY_REPR_ORDER.get(entity_type, [])
    present = set(df["representation"].unique()) if "representation" in df.columns else set()
    ordered = [r for r in canonical if r in present]
    extras = sorted(present - set(canonical))
    return ordered + extras


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_summary_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["entity_type", "representation"] + [m for m in METRICS if m in df.columns]
    out = df[[c for c in cols if c in df.columns]].copy()
    for m in METRICS:
        if m in out.columns:
            out[m] = out[m].apply(lambda x: f"{float(x):.4f}")
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def write_summary_json(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in df.iterrows():
        record: dict = {
            "entity_type": str(row.get("entity_type", "")),
            "representation": str(row["representation"]),
        }
        for m in METRICS:
            if m in row:
                record[m] = round(float(row[m]), 4)
        records.append(record)
    path.write_text(
        json.dumps({"rows": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort: entity type canonical order, then NDCG descending within each group
    df = df.copy()
    df["_et"] = df.get("entity_type", pd.Series(dtype=str)).map(_entity_order_key).fillna(99)
    ndcg_col = "NDCG" if "NDCG" in df.columns else METRICS[-1]
    df = df.sort_values(["_et", ndcg_col], ascending=[True, False]).drop(columns=["_et"])

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{Pre-Retrieval Representation Evaluation Metrics (500 questions, top-10 retrieval)}",
        r"  \label{tab:pre_retrieval_representation_metrics}",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    \textbf{Entity Type} & \textbf{Representation} & "
        r"\textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{MRR} & \textbf{NDCG} \\",
        r"    \midrule",
    ]

    prev_entity = None
    for _, row in df.iterrows():
        et = str(row.get("entity_type", ""))
        repr_name = str(row.get("representation", ""))
        if et != prev_entity:
            if prev_entity is not None:
                lines.append(r"    \midrule")
            lines.append(
                f"    \\multicolumn{{7}}{{l}}"
                f"{{\\textit{{{_latex_escape(et.capitalize())}}}}} \\\\"
            )
            prev_entity = et
        vals = " & ".join(
            f"{float(row[m]):.4f}" if m in row else "---"
            for m in METRICS
        )
        lines.append(f"    {_latex_escape(et)} & {_latex_escape(repr_name)} & {vals} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Group A — Overall figures (horizontal bar, one panel per entity type)
# ---------------------------------------------------------------------------

def plot_overall(df: pd.DataFrame, output_dir: Path, metric: str) -> Path:
    metric_slug = METRIC_FILE_NAMES.get(metric, metric.lower().replace("@", "_at_"))
    out_path = output_dir / f"overall_{metric_slug}_by_representation.pdf"

    if df.empty or metric not in df.columns:
        print(f"  WARNING: no data for overall {metric} figure — skipping", file=sys.stderr)
        return out_path

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(f"Pre-Retrieval Evaluation — Overall {metric} by Representation", fontsize=12)

        for ax, entity_type in zip(axes, ENTITY_TYPES):
            sub = _filter_entity(df, entity_type)
            if sub.empty:
                ax.set_title(entity_type.capitalize(), fontweight="bold")
                ax.set_visible(False)
                continue

            sub = sub.sort_values(metric, ascending=True)  # ascending → best bar is at top in horizontal
            labels = [_short(r) for r in sub["representation"]]
            values = sub[metric].values
            color = ENTITY_COLORS.get(entity_type, "#4C72B0")

            bars = ax.barh(labels, values, color=color, edgecolor="white", linewidth=0.5)
            ax.set_xlim(0, min(1.0, max(values) * 1.18 + 0.05))
            ax.set_xlabel(metric)
            ax.set_title(entity_type.capitalize(), fontweight="bold")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar, val in zip(bars, values):
                ax.text(
                    val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8,
                )

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Group B — By difficulty (grouped vertical bars, one panel per entity type)
# ---------------------------------------------------------------------------

def plot_by_difficulty(df_diff: pd.DataFrame, output_dir: Path, metric: str) -> Path:
    metric_slug = METRIC_FILE_NAMES.get(metric, metric.lower().replace("@", "_at_"))
    out_path = output_dir / f"difficulty_{metric_slug}_by_representation.pdf"

    if df_diff.empty or "difficulty" not in df_diff.columns or metric not in df_diff.columns:
        print(f"  WARNING: no difficulty data for {metric} — skipping", file=sys.stderr)
        return out_path

    difficulties = [d for d in DIFFICULTY_ORDER if d in df_diff["difficulty"].unique()]
    bar_width = 0.75 / max(len(difficulties), 1)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(f"Pre-Retrieval Evaluation — {metric} by Difficulty", fontsize=12)

        for ax, entity_type in zip(axes, ENTITY_TYPES):
            sub = _filter_entity(df_diff, entity_type)
            if sub.empty:
                ax.set_title(entity_type.capitalize(), fontweight="bold")
                continue

            repr_list = _ordered_reprs(sub, entity_type)
            if not repr_list:
                continue

            x = np.arange(len(repr_list))
            n = len(difficulties)
            for i, diff in enumerate(difficulties):
                diff_sub = sub[sub["difficulty"] == diff]
                vals = [
                    float(diff_sub.loc[diff_sub["representation"] == r, metric].values[0])
                    if r in diff_sub["representation"].values else 0.0
                    for r in repr_list
                ]
                offset = (i - n / 2 + 0.5) * bar_width
                ax.bar(
                    x + offset, vals, width=bar_width * 0.9,
                    label=diff, color=DIFFICULTY_COLORS.get(diff, "#999"),
                    edgecolor="white", linewidth=0.3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([_short(r) for r in repr_list], rotation=35, ha="right")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(metric)
            ax.set_title(entity_type.capitalize(), fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if entity_type == ENTITY_TYPES[0]:
                ax.legend(title="Difficulty", loc="upper left")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Group C — By category (grouped vertical bars, one panel per entity type)
# ---------------------------------------------------------------------------

def plot_by_category(df_cat: pd.DataFrame, output_dir: Path, metric: str) -> Path:
    metric_slug = METRIC_FILE_NAMES.get(metric, metric.lower().replace("@", "_at_"))
    out_path = output_dir / f"category_{metric_slug}_by_representation.pdf"

    if df_cat.empty or "category" not in df_cat.columns or metric not in df_cat.columns:
        print(f"  WARNING: no category data for {metric} — skipping", file=sys.stderr)
        return out_path

    categories = [c for c in CATEGORY_ORDER if c in df_cat["category"].unique()]
    if not categories:
        print(f"  WARNING: no recognised categories in data — skipping {metric} category figure", file=sys.stderr)
        return out_path

    bar_width = 0.75 / max(len(categories), 1)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(f"Pre-Retrieval Evaluation — {metric} by Question Category", fontsize=12)

        for ax, entity_type in zip(axes, ENTITY_TYPES):
            sub = _filter_entity(df_cat, entity_type)
            if sub.empty:
                ax.set_title(entity_type.capitalize(), fontweight="bold")
                continue

            repr_list = _ordered_reprs(sub, entity_type)
            if not repr_list:
                continue

            # Only show a category if this entity type actually evaluated its questions.
            # summary_by_category.json carries zero-metric cross-entity rows (e.g. model
            # representations appear in the "paper" segment with evaluated_questions=0);
            # those must be excluded so panels stay clean.
            if "evaluated_questions" in sub.columns:
                present_cats = [
                    c for c in categories
                    if c in sub["category"].unique()
                    and (sub.loc[sub["category"] == c, "evaluated_questions"] > 0).any()
                ]
            else:
                present_cats = [c for c in categories if c in sub["category"].unique()]
            x = np.arange(len(repr_list))
            n = len(present_cats)
            for i, cat in enumerate(present_cats):
                cat_sub = sub[sub["category"] == cat]
                vals = [
                    float(cat_sub.loc[cat_sub["representation"] == r, metric].values[0])
                    if r in cat_sub["representation"].values else 0.0
                    for r in repr_list
                ]
                offset = (i - n / 2 + 0.5) * bar_width
                ax.bar(
                    x + offset, vals, width=bar_width * 0.9,
                    label=cat, color=CATEGORY_COLORS.get(cat, "#999"),
                    edgecolor="white", linewidth=0.3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([_short(r) for r in repr_list], rotation=35, ha="right")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(metric)
            ax.set_title(entity_type.capitalize(), fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if entity_type == ENTITY_TYPES[0]:
                ax.legend(title="Category", loc="upper left", ncol=2)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pre-retrieval result artifacts (CSV, JSON, LaTeX, PDF figures)."
    )
    parser.add_argument(
        "--input-dir",
        default="data/results/pre_retrieval",
        help="Directory containing pre-retrieval results (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for artifacts (defaults to --input-dir).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading data...")
    df_overall = load_summary(input_dir)
    df_difficulty = load_by_segment(input_dir, "summary_by_difficulty.json")
    df_category = load_by_segment(input_dir, "summary_by_category.json")

    if df_overall.empty:
        print(
            "ERROR: no summary data found. Run pre-retrieval evaluations first.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_total = len(df_overall)
    n_et = df_overall["entity_type"].nunique() if "entity_type" in df_overall.columns else "?"
    print(f"  {n_total} representation rows across {n_et} entity types")
    print(f"  difficulty rows : {len(df_difficulty)}")
    print(f"  category rows   : {len(df_category)}")
    print()

    generated: list[Path] = []

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    print("Writing summaries...")

    csv_path = output_dir / "summary_all_representations.csv"
    write_summary_csv(df_overall, csv_path)
    generated.append(csv_path)

    json_path = output_dir / "summary_all_representations.json"
    write_summary_json(df_overall, json_path)
    generated.append(json_path)

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    print("Writing LaTeX table...")
    tex_path = output_dir / "thesis_tables" / "pre_retrieval_representation_metrics.tex"
    write_latex_table(df_overall, tex_path)
    generated.append(tex_path)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    for metric in FIGURE_METRICS:
        generated.append(plot_overall(df_overall, figures_dir, metric))
        if not df_difficulty.empty:
            generated.append(plot_by_difficulty(df_difficulty, figures_dir, metric))
        else:
            print(f"  Skipping difficulty figures for {metric} (no data)")
        if not df_category.empty:
            generated.append(plot_by_category(df_category, figures_dir, metric))
        else:
            print(f"  Skipping category figures for {metric} (no data)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("Generated files:")
    for p in generated:
        exists = "✓" if p.exists() else "✗ (skipped)"
        print(f"  {exists}  {p}")


if __name__ == "__main__":
    main()
