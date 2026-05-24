#!/usr/bin/env python3
"""
generate_chapter4_figures.py

Generates Chapter 4 thesis figures from current (500-question) evaluation results.

Outputs files with _current suffix to docs/chapter4_artifacts/figures/.
Never touches existing non-_current files.

Usage:
    python scripts/generate_chapter4_figures.py
    python scripts/generate_chapter4_figures.py --force   # regenerate even if exists

Figures produced:
    fig_pre_retrieval_ndcg_by_representation_current.{pdf,png}
    fig_pre_retrieval_difficulty_ndcg_current.{pdf,png}
    fig_retrieval_method_comparison_current.{pdf,png}
    fig_retrieval_hit_gap_current.{pdf,png}
    fig_retrieval_question_type_heatmap_current.{pdf,png}
    fig_retrieval_by_entity_type_current.{pdf,png}
    fig_retrieval_by_difficulty_current.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_PRE_SUMMARY  = REPO_ROOT / "data" / "results" / "pre_retrieval" / "summary.csv"
SRC_PRE_DIFF_JSON = REPO_ROOT / "data" / "results" / "pre_retrieval" / "summary_by_difficulty.json"
SRC_RET_SUMMARY  = REPO_ROOT / "data" / "results" / "retrieval" / "summary.csv"
SRC_RET_ENTITY   = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_entity_type_ndcg.csv"
SRC_RET_DIFF     = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_difficulty_ndcg.csv"
SRC_RET_QTYPE    = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_question_type_ndcg.csv"
SRC_RET_GAP      = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_precision_recall_tradeoff.csv"

OUT_DIR = REPO_ROOT / "docs" / "chapter4_artifacts" / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DPI = 300
FONT_SIZE = 9

# Entity colours
C_PAPER   = "#4472C4"
C_DATASET = "#E07B54"
C_MODEL   = "#70A45E"
ENTITY_COLORS = {"paper": C_PAPER, "dataset": C_DATASET, "model": C_MODEL}

# Method colours: baseline grey, hybrid blue shades, RRF amber
C_BASELINE = "#AAAAAA"
C_HYBRID   = "#4472C4"
C_RRF      = "#F28E2B"
# Per-method colour for ordered list
METHOD_COLORS = [C_BASELINE, "#6A8DC8", "#5578B5", "#4060A0", C_RRF, "#D4720A"]

C_BEST  = "#1A3A6B"
C_OTHER = "#A8BDD8"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FONT_SIZE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    FONT_SIZE + 1,
    "axes.labelsize":    FONT_SIZE,
    "xtick.labelsize":   FONT_SIZE - 1,
    "ytick.labelsize":   FONT_SIZE - 1,
    "legend.fontsize":   FONT_SIZE - 1,
    "figure.dpi":        100,
})

# ---------------------------------------------------------------------------
# Method labels
# ---------------------------------------------------------------------------

METHOD_ORDER = [
    "pure_semantic_dense",
    "hybrid_type_filtering",
    "hybrid_type_onehop_filtering",
    "hybrid_predicate_aware_filtering",
    "optional_rrf_fusion",
    "optional_rrf_symbolic",
]

METHOD_SHORT = {
    "pure_semantic_dense":               "Dense",
    "hybrid_type_filtering":             "Hybrid-Type",
    "hybrid_type_onehop_filtering":      "Hybrid-OneHop",
    "hybrid_predicate_aware_filtering":  "Hybrid-Pred.",
    "optional_rrf_fusion":               "RRF-Fusion",
    "optional_rrf_symbolic":             "RRF-Symbolic",
}

REPR_DISPLAY = {
    "title_only":                  "Title Only",
    "abstract_only":               "Abstract Only",
    "title_abstract":              "Title + Abstract",
    "predicate_filtered":          "Predicate-Filtered",
    "enriched_metadata":           "Enriched Metadata",
    "one_hop":                     "One-Hop",
    "dataset_title_only":          "Title Only",
    "dataset_metadata":            "Metadata",
    "dataset_predicate_filtered":  "Predicate-Filtered",
    "dataset_enriched_metadata":   "Enriched Metadata",
    "model_title_only":            "Title Only",
    "model_metadata":              "Metadata",
    "model_predicate_filtered":    "Predicate-Filtered",
    "model_enriched_metadata":     "Enriched Metadata",
}

QTYPE_DISPLAY = {
    "paper_title_to_entity":       "Paper: Title→Entity",
    "paper_abstract_semantic":     "Paper: Abstract Semantic",
    "paper_description_to_title":  "Paper: Desc.→Title",
    "paper_multi_metadata":        "Paper: Multi-Metadata",
    "paper_author_to_paper":       "Paper: Author→Paper",
    "paper_task_to_paper":         "Paper: Task→Paper",
    "dataset_title_to_entity":     "Dataset: Title→Entity",
    "dataset_task_to_dataset":     "Dataset: Task→Dataset",
    "dataset_multi_metadata":      "Dataset: Multi-Metadata",
    "dataset_keyword_to_dataset":  "Dataset: Keyword",
    "model_family_variant":        "Model: Family Variant",
    "model_repository_to_model":   "Model: Repo→Model",
    "model_name_to_entity":        "Model: Name→Entity",
    "model_multi_metadata":        "Model: Multi-Metadata",
    "model_paper_to_model":        "Model: Paper→Model",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

generated: list[tuple[str, Path]] = []
skipped_files: list[tuple[str, Path]] = []


def validate():
    required = [SRC_PRE_SUMMARY, SRC_PRE_DIFF_JSON, SRC_RET_SUMMARY,
                SRC_RET_ENTITY, SRC_RET_DIFF, SRC_RET_QTYPE, SRC_RET_GAP]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("ERROR: Missing source files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


def save_figure(fig: plt.Figure, stem: str, force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    already_exists = pdf.exists() and png.exists()
    if already_exists and not force:
        plt.close(fig)
        skipped_files.append((stem, pdf))
        print(f"  [SKIP]  {stem}  (already exists; use --force to overwrite)")
        return
    fig.savefig(pdf, bbox_inches="tight", format="pdf")
    fig.savefig(png, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    generated.append((stem, pdf))
    print(f"  [FIGURE] {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# FIGURE 1 — Pre-retrieval NDCG by representation (3-panel horizontal bar)
# ---------------------------------------------------------------------------

def make_fig_pre_retrieval_ndcg(force: bool) -> None:
    df = pd.read_csv(SRC_PRE_SUMMARY)
    df.columns = df.columns.str.strip()

    entity_order = ["paper", "dataset", "model"]
    entity_labels = {"paper": "Papers (250 questions)",
                     "dataset": "Datasets (125 questions)",
                     "model": "Models (125 questions)"}

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))

    for ax, et in zip(axes, entity_order):
        group = df[df["entity_type"].str.strip() == et].copy()
        group["label"] = group["representation"].map(
            lambda r: REPR_DISPLAY.get(r, r.replace("_", " ").title())
        )
        group = group.sort_values("NDCG", ascending=True).reset_index(drop=True)
        best_ndcg = float(group["NDCG"].max())

        colors = [C_BEST if abs(float(v) - best_ndcg) < 1e-9 else C_OTHER
                  for v in group["NDCG"]]

        ax.barh(group["label"], group["NDCG"].astype(float), color=colors,
                height=0.55, edgecolor="white", linewidth=0.3)

        # Annotate each bar
        for idx, (_, row) in enumerate(group.iterrows()):
            val = float(row["NDCG"])
            color = "white" if abs(val - best_ndcg) < 1e-9 else "#444444"
            ax.text(val - 0.008, idx, f"{val:.4f}",
                    va="center", ha="right", fontsize=FONT_SIZE - 2, color=color)

        ax.set_xlim(0, min(1.0, best_ndcg * 1.25 + 0.04))
        ax.set_xlabel("NDCG")
        ax.set_ylabel(entity_labels[et], fontsize=FONT_SIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)

    best_patch  = mpatches.Patch(color=C_BEST,  label="Best strategy per entity type")
    other_patch = mpatches.Patch(color=C_OTHER, label="Other strategies")
    fig.legend(handles=[best_patch, other_patch],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005),
               fontsize=FONT_SIZE - 1, frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "fig_pre_retrieval_ndcg_by_representation_current", force)


# ---------------------------------------------------------------------------
# FIGURE 2 — Pre-retrieval NDCG by difficulty (grouped bars, best repr only)
# ---------------------------------------------------------------------------

def make_fig_pre_retrieval_difficulty(force: bool) -> None:
    # Determine best representations from summary.csv
    df_summary = pd.read_csv(SRC_PRE_SUMMARY)
    df_summary.columns = df_summary.columns.str.strip()
    best_repr = {}
    for et in ["paper", "dataset", "model"]:
        group = df_summary[df_summary["entity_type"].str.strip() == et]
        best_repr[et] = str(group.loc[group["NDCG"].idxmax(), "representation"]).strip()

    # Load difficulty data
    raw = json.load(open(SRC_PRE_DIFF_JSON, encoding="utf-8"))
    rows = []
    for diff, seg in raw.get("segments", {}).items():
        for r in seg.get("rows", []):
            rows.append(r)
    df_diff = pd.DataFrame(rows)
    df_diff.columns = df_diff.columns.str.strip()

    diff_order   = ["easy", "medium", "hard"]
    entity_order = ["paper", "dataset", "model"]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    n_diffs    = len(diff_order)
    n_entities = len(entity_order)
    bar_width  = 0.22
    x = np.arange(n_diffs)

    for i, et in enumerate(entity_order):
        rep = best_repr[et]
        sub = df_diff[
            (df_diff["entity_type"].str.strip() == et) &
            (df_diff["representation"].str.strip() == rep)
        ]

        vals = []
        for d in diff_order:
            row = sub[sub["difficulty"].str.strip() == d]
            vals.append(float(row.iloc[0]["NDCG"]) if not row.empty else np.nan)

        rep_label = REPR_DISPLAY.get(rep, rep.replace("_", " ").title())
        offset = (i - n_entities / 2 + 0.5) * bar_width
        color  = ENTITY_COLORS[et]

        for j, (xval, yval) in enumerate(zip(x + offset, vals)):
            if not np.isnan(yval):
                label = f"{et.title()} ({rep_label})" if j == 0 else "_nolegend_"
                ax.bar(xval, yval, width=bar_width, color=color,
                       label=label, alpha=0.88, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy\n(131 q)", "Medium\n(177 q)", "Hard\n(192 q)"])
    ax.set_ylabel("NDCG")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE - 1, frameon=False,
              bbox_to_anchor=(0.5, -0.22), loc="upper center", ncol=3)

    fig.tight_layout()
    save_figure(fig, "fig_pre_retrieval_difficulty_ndcg_current", force)


# ---------------------------------------------------------------------------
# FIGURE 3 — Retrieval method comparison (NDCG + Hit@10 grouped bars)
# ---------------------------------------------------------------------------

def make_fig_retrieval_method_comparison(force: bool) -> None:
    df = pd.read_csv(SRC_RET_SUMMARY)
    df.columns = df.columns.str.strip()

    # Preserve canonical method order
    df["_order"] = df["method"].str.strip().map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    short_labels = [METHOD_SHORT.get(str(m).strip(), str(m)) for m in df["method"]]
    x = np.arange(len(df))
    bar_width = 0.33

    fig, ax = plt.subplots(figsize=(7.5, 4))

    for i, (_, row) in enumerate(df.iterrows()):
        method = str(row["method"]).strip()
        is_base = method == "pure_semantic_dense"
        c_ndcg  = C_BASELINE if is_base else C_HYBRID if "hybrid" in method else C_RRF
        c_hit10 = C_BASELINE if is_base else "#7BA8E0" if "hybrid" in method else "#F5B86B"

        ax.bar(x[i] - bar_width / 2, float(row["NDCG"]),  bar_width,
               color=c_ndcg,  alpha=0.9, edgecolor="white",
               label="NDCG"   if i == 0 else "_nolegend_")
        ax.bar(x[i] + bar_width / 2, float(row["Hit@10"]), bar_width,
               color=c_hit10, alpha=0.9, edgecolor="white",
               label="Hit@10" if i == 0 else "_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.80)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Custom legend
    ndcg_patch  = mpatches.Patch(color=C_HYBRID,   label="NDCG (non-baseline)")
    hit10_patch = mpatches.Patch(color="#7BA8E0",  label="Hit@10 (non-baseline)")
    base_patch  = mpatches.Patch(color=C_BASELINE, label="Dense baseline")
    rrf_patch   = mpatches.Patch(color=C_RRF,      label="NDCG (RRF)")
    ax.legend(handles=[base_patch, ndcg_patch, hit10_patch, rrf_patch],
              fontsize=FONT_SIZE - 1, frameon=False, ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_method_comparison_current", force)


# ---------------------------------------------------------------------------
# FIGURE 4 — Hit@1 vs Hit@10 recoverability gap
# ---------------------------------------------------------------------------

def make_fig_retrieval_hit_gap(force: bool) -> None:
    df = pd.read_csv(SRC_RET_GAP)
    df.columns = df.columns.str.strip()

    df["_order"] = df["method"].str.strip().map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    short_labels = [METHOD_SHORT.get(str(m).strip(), str(m)) for m in df["method"]]
    gap_col = "gap_Hit10_minus_Hit1"

    x = np.arange(len(df))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4))

    for i, (_, row) in enumerate(df.iterrows()):
        method = str(row["method"]).strip()
        is_base = method == "pure_semantic_dense"
        c_h1  = C_BASELINE if is_base else (C_RRF   if "rrf" in method else C_HYBRID)
        c_h10 = C_BASELINE if is_base else ("#F5B86B" if "rrf" in method else "#7BA8E0")

        ax.bar(x[i] - bar_width / 2, float(row["Hit@1"]),  bar_width,
               color=c_h1,  alpha=0.9, edgecolor="white",
               label="Hit@1"  if i == 1 else "_nolegend_")
        ax.bar(x[i] + bar_width / 2, float(row["Hit@10"]), bar_width,
               color=c_h10, alpha=0.9, edgecolor="white",
               label="Hit@10" if i == 1 else "_nolegend_")

        gap = float(row[gap_col])
        top = max(float(row["Hit@1"]), float(row["Hit@10"])) + 0.018
        ax.annotate(f"+{gap:.3f}", xy=(x[i], top),
                    ha="center", va="bottom",
                    fontsize=FONT_SIZE - 2, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    h1_patch  = mpatches.Patch(color=C_HYBRID,    label="Hit@1 (hybrid)")
    h10_patch = mpatches.Patch(color="#7BA8E0",   label="Hit@10 (hybrid)")
    base_patch = mpatches.Patch(color=C_BASELINE, label="Dense baseline")
    rrf_h1    = mpatches.Patch(color=C_RRF,       label="Hit@1 (RRF)")
    rrf_h10   = mpatches.Patch(color="#F5B86B",   label="Hit@10 (RRF)")
    ax.legend(handles=[base_patch, h1_patch, h10_patch, rrf_h1, rrf_h10],
              fontsize=FONT_SIZE - 1, frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.22), loc="upper center")

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_hit_gap_current", force)


# ---------------------------------------------------------------------------
# FIGURE 5 — Question-type NDCG heatmap
# ---------------------------------------------------------------------------

def make_fig_retrieval_question_type_heatmap(force: bool) -> None:
    df = pd.read_csv(SRC_RET_QTYPE)
    df.columns = df.columns.str.strip()

    # Filter: N >= 3
    max_counts = df.groupby("question_type")["count"].max()
    valid_qtypes = max_counts[max_counts >= 3].index.tolist()

    # Sort by dense NDCG descending
    dense_ndcg = (df[df["method"].str.strip() == "pure_semantic_dense"]
                  .set_index("question_type")["NDCG"])
    valid_qtypes = sorted(valid_qtypes, key=lambda q: -float(dense_ndcg.get(q, 0)))

    methods = [m for m in METHOD_ORDER if m in df["method"].str.strip().unique()]

    data = np.full((len(valid_qtypes), len(methods)), np.nan)
    for ri, qt in enumerate(valid_qtypes):
        for ci, m in enumerate(methods):
            sub = df[(df["question_type"].str.strip() == qt) &
                     (df["method"].str.strip() == m)]
            if not sub.empty:
                data[ri, ci] = float(sub.iloc[0]["NDCG"])

    # Add count column info to y-labels
    qt_labels = []
    for qt in valid_qtypes:
        n = int(max_counts[qt])
        label = QTYPE_DISPLAY.get(qt, qt.replace("_", " ").title())
        qt_labels.append(f"{label}  (N={n})")

    short_methods = [METHOD_SHORT.get(m, m) for m in methods]

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    fig_h = max(5.5, len(valid_qtypes) * 0.46)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(short_methods)))
    ax.set_xticklabels(short_methods, rotation=30, ha="right", fontsize=FONT_SIZE)
    ax.set_yticks(range(len(qt_labels)))
    ax.set_yticklabels(qt_labels, fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Retrieval Method", fontsize=FONT_SIZE)
    ax.set_ylabel("Question Type", fontsize=FONT_SIZE)

    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val = data[ri, ci]
            if not np.isnan(val):
                nv = norm(val)
                txt_color = "black" if 0.30 < nv < 0.80 else "white"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=FONT_SIZE - 2, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("NDCG", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_question_type_heatmap_current", force)


# ---------------------------------------------------------------------------
# FIGURE 6 — Retrieval NDCG by entity type (heatmap)
# ---------------------------------------------------------------------------

def make_fig_retrieval_by_entity_type(force: bool) -> None:
    df = pd.read_csv(SRC_RET_ENTITY)
    df.columns = df.columns.str.strip()

    df["_order"] = df["method"].str.strip().map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    cols   = ["paper_NDCG", "dataset_NDCG", "model_NDCG"]
    labels = ["Paper", "Dataset", "Model"]
    data   = df[cols].values.astype(float)
    y_labs = [METHOD_SHORT.get(str(m).strip(), str(m)) for m in df["method"]]

    vmin, vmax = float(data.min()), float(data.max())
    norm = Normalize(vmin=vmin - 0.01, vmax=vmax + 0.01)
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_yticks(range(len(y_labs)))
    ax.set_yticklabels(y_labs, fontsize=FONT_SIZE)
    ax.set_xlabel("Entity Type", fontsize=FONT_SIZE)
    ax.set_ylabel("Retrieval Method", fontsize=FONT_SIZE)

    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val = data[ri, ci]
            nv  = norm(val)
            txt_color = "white" if nv > 0.65 else "black"
            ax.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                    fontsize=FONT_SIZE - 1, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("NDCG", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_by_entity_type_current", force)


# ---------------------------------------------------------------------------
# FIGURE 7 — Retrieval NDCG by difficulty (grouped bars)
# ---------------------------------------------------------------------------

def make_fig_retrieval_by_difficulty(force: bool) -> None:
    df = pd.read_csv(SRC_RET_DIFF)
    df.columns = df.columns.str.strip()

    df["_order"] = df["method"].str.strip().map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    diff_cols   = ["easy_NDCG", "medium_NDCG", "hard_NDCG"]
    diff_labels = ["Easy\n(131 q)", "Medium\n(177 q)", "Hard\n(192 q)"]
    methods = df["method"].str.strip().tolist()
    n       = len(methods)
    bar_w   = 0.13
    x       = np.arange(len(diff_cols))

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (_, row) in enumerate(df.iterrows()):
        vals   = [float(row[c]) for c in diff_cols]
        offset = (i - n / 2 + 0.5) * bar_w
        color  = METHOD_COLORS[i % len(METHOD_COLORS)]
        short  = METHOD_SHORT.get(str(row["method"]).strip(), "")
        ax.bar(x + offset, vals, bar_w, label=short, color=color,
               alpha=0.88, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.set_ylabel("NDCG")
    ax.set_ylim(0, 0.88)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE - 1, ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.22), loc="upper center")

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_by_difficulty_current", force)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 figures from current (500-question) results."
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing _current files.")
    args = parser.parse_args()

    print("=" * 60)
    print("Chapter 4 Figure Generator  (500-question dataset)")
    print("=" * 60)

    print("\nValidating source files ...")
    validate()
    print("  All source files present.")

    print("\nGenerating figures ...")
    make_fig_pre_retrieval_ndcg(args.force)
    make_fig_pre_retrieval_difficulty(args.force)
    make_fig_retrieval_method_comparison(args.force)
    make_fig_retrieval_hit_gap(args.force)
    make_fig_retrieval_question_type_heatmap(args.force)
    make_fig_retrieval_by_entity_type(args.force)
    make_fig_retrieval_by_difficulty(args.force)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if generated:
        print(f"\nGenerated ({len(generated)}):")
        for name, path in generated:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}")
    if skipped_files:
        print(f"\nSkipped ({len(skipped_files)}) — already current:")
        for name, path in skipped_files:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}")
    if not generated and not skipped_files:
        print("  Nothing to do.")


if __name__ == "__main__":
    main()
