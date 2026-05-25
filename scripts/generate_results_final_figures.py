#!/usr/bin/env python3
"""
generate_results_final_figures.py

Regenerates all final thesis-ready Chapter 4 figures from the latest raw result files.
Outputs to figs/results_final/.  Safe to rerun (--force to overwrite).

Key design choices vs. the older generate_chapter4_figures.py:
  - Output location: figs/results_final/  (canonical, versioned)
  - Figure B shows all 6 methods; hybrid_type_filtering is still a no-op (≡ Dense)
    and is hatched; predicate-aware and RRF-Symbolic are now behaviourally distinct
  - Figures C / D / E show the 5 behaviourally distinct methods
    (Dense, OneHop, Predicate, RRF-Fusion, RRF-Symbolic); Hybrid-Type omitted
  - Figure A uses per-entity-type colour for the best-representation bar

Usage:
    py scripts/generate_results_final_figures.py
    py scripts/generate_results_final_figures.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

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

SRC_PRE_ALL  = REPO_ROOT / "data" / "results" / "pre_retrieval" / "summary_all_representations.csv"
SRC_PRE_DIFF = REPO_ROOT / "data" / "results" / "pre_retrieval" / "summary_by_difficulty.json"
SRC_RET      = REPO_ROOT / "data" / "results" / "retrieval" / "summary.csv"
SRC_ENTITY   = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_entity_type_ndcg.csv"
SRC_DIFF     = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_difficulty_ndcg.csv"
SRC_QTYPE    = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_question_type_ndcg.csv"
SRC_GAP      = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_precision_recall_tradeoff.csv"

OUT_DIR = REPO_ROOT / "figs" / "results_final"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DPI = 300

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        100,
})

FS = 9   # base font size alias

# Entity colours (best bar in panel)
C_PAPER   = "#4472C4"
C_DATASET = "#D06030"
C_MODEL   = "#4E8A40"
# Lighter shades for non-best bars in each panel
C_PAPER_L   = "#B8C9E8"
C_DATASET_L = "#EFC0A8"
C_MODEL_L   = "#B4D4AC"

ENTITY_COLORS     = {"paper": C_PAPER,   "dataset": C_DATASET,   "model": C_MODEL}
ENTITY_COLORS_LT  = {"paper": C_PAPER_L, "dataset": C_DATASET_L, "model": C_MODEL_L}

# Retrieval method colours
C_BASELINE = "#AAAAAA"
C_HYBRID   = "#4472C4"
C_RRF      = "#F28E2B"
C_HIT      = "#7BA8E0"
C_RRF_HIT  = "#F5B86B"

# Methods
METHOD_ORDER = [
    "pure_semantic_dense",
    "hybrid_type_filtering",
    "hybrid_type_onehop_filtering",
    "hybrid_predicate_aware_filtering",
    "optional_rrf_fusion",
    "optional_rrf_symbolic",
]

NOOP_METHODS = {
    "hybrid_type_filtering",
    # hybrid_predicate_aware_filtering and optional_rrf_symbolic were previously
    # no-ops due to stale question-type names in filtering.py.  After the fix
    # they produce distinct rankings and are no longer annotated as no-ops.
}

DISTINCT_METHODS = [
    "pure_semantic_dense",
    "hybrid_type_onehop_filtering",
    "hybrid_predicate_aware_filtering",
    "optional_rrf_fusion",
    "optional_rrf_symbolic",
]

NOOP_LABEL = {
    "hybrid_type_filtering": "≡ Dense",
}

METHOD_SHORT = {
    "pure_semantic_dense":               "Dense",
    "hybrid_type_filtering":             "Hybrid-Type",
    "hybrid_type_onehop_filtering":      "Hybrid-OneHop",
    "hybrid_predicate_aware_filtering":  "Hybrid-Pred.",
    "optional_rrf_fusion":               "RRF-Fusion",
    "optional_rrf_symbolic":             "RRF-Symbolic",
}

REPR_DISPLAY = {
    "title_only":                 "Title Only",
    "abstract_only":              "Abstract Only",
    "title_abstract":             "Title + Abstract",
    "predicate_filtered":         "Predicate-Filtered",
    "enriched_metadata":          "Enriched Metadata",
    "one_hop":                    "One-Hop",
    "dataset_title_only":         "Title Only",
    "dataset_metadata":           "Metadata",
    "dataset_predicate_filtered": "Predicate-Filtered",
    "dataset_enriched_metadata":  "Enriched Metadata",
    "model_title_only":           "Title Only",
    "model_metadata":             "Metadata",
    "model_predicate_filtered":   "Predicate-Filtered",
    "model_enriched_metadata":    "Enriched Metadata",
}

QTYPE_DISPLAY = {
    "paper_title_to_entity":      "Paper: Title→Entity",
    "paper_abstract_semantic":    "Paper: Abstract Semantic",
    "paper_description_to_title": "Paper: Desc.→Title",
    "paper_multi_metadata":       "Paper: Multi-Metadata",
    "paper_author_to_paper":      "Paper: Author→Paper",
    "paper_task_to_paper":        "Paper: Task→Paper",
    "dataset_title_to_entity":    "Dataset: Title→Entity",
    "dataset_task_to_dataset":    "Dataset: Task→Dataset",
    "dataset_multi_metadata":     "Dataset: Multi-Metadata",
    "dataset_keyword_to_dataset": "Dataset: Keyword",
    "model_family_variant":       "Model: Family Variant",
    "model_repository_to_model":  "Model: Repo→Model",
    "model_name_to_entity":       "Model: Name→Entity",
    "model_multi_metadata":       "Model: Multi-Metadata",
    "model_paper_to_model":       "Model: Paper→Model",
}

# ---------------------------------------------------------------------------
# Thesis value reference (for validation)
# ---------------------------------------------------------------------------

THESIS_VALUES = {
    "paper_best_repr":    ("predicate_filtered", 0.4544),
    "dataset_best_repr":  ("dataset_metadata",   0.5281),
    "model_best_repr":    ("model_metadata",      0.5046),
    "dense_ndcg":         0.4854,
    "dense_hit10":        0.598,
    "rrf_fusion_ndcg":    0.4962,
    "rrf_fusion_hit10":   0.618,
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

generated:    list[tuple[str, Path]] = []
skipped_list: list[tuple[str, Path]] = []
warnings_log: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, stem: str, force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    if pdf.exists() and png.exists() and not force:
        plt.close(fig)
        skipped_list.append((stem, pdf))
        print(f"  [SKIP]  {stem}  (use --force to overwrite)")
        return
    fig.savefig(pdf, bbox_inches="tight", format="pdf")
    fig.savefig(png, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    generated.append((stem, pdf))
    print(f"  [OK]    {stem}.pdf / .png")


def _warn(msg: str) -> None:
    warnings_log.append(msg)
    print(f"  [WARN]  {msg}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate() -> None:
    sources = [SRC_PRE_ALL, SRC_PRE_DIFF, SRC_RET,
               SRC_ENTITY, SRC_DIFF, SRC_QTYPE, SRC_GAP]
    missing = [p for p in sources if not p.exists()]
    if missing:
        print("ERROR: Missing source files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Check thesis values against raw data
    pre_df = pd.read_csv(SRC_PRE_ALL)
    pre_df.columns = pre_df.columns.str.strip()

    for et, (exp_repr, exp_ndcg) in [
        ("paper",   THESIS_VALUES["paper_best_repr"]),
        ("dataset", THESIS_VALUES["dataset_best_repr"]),
        ("model",   THESIS_VALUES["model_best_repr"]),
    ]:
        group = pre_df[pre_df["entity_type"].str.strip() == et]
        best_row = group.loc[group["NDCG"].idxmax()]
        actual_repr = str(best_row["representation"]).strip()
        actual_ndcg = float(best_row["NDCG"])
        if actual_repr != exp_repr or abs(actual_ndcg - exp_ndcg) > 1e-4:
            _warn(
                f"Pre-retrieval {et}: thesis claims best={exp_repr} NDCG={exp_ndcg:.4f}, "
                f"data shows best={actual_repr} NDCG={actual_ndcg:.4f}"
            )

    ret_df = pd.read_csv(SRC_RET)
    ret_df.columns = ret_df.columns.str.strip()
    ret_df["method"] = ret_df["method"].str.strip()

    for method_key, metric, exp_val in [
        ("pure_semantic_dense", "NDCG",   THESIS_VALUES["dense_ndcg"]),
        ("pure_semantic_dense", "Hit@10", THESIS_VALUES["dense_hit10"]),
        ("optional_rrf_fusion", "NDCG",   THESIS_VALUES["rrf_fusion_ndcg"]),
        ("optional_rrf_fusion", "Hit@10", THESIS_VALUES["rrf_fusion_hit10"]),
    ]:
        row = ret_df[ret_df["method"] == method_key]
        if row.empty:
            _warn(f"Method {method_key} not found in retrieval summary")
            continue
        actual = float(row.iloc[0][metric])
        if abs(actual - exp_val) > 1e-4:
            _warn(
                f"Retrieval {method_key} {metric}: thesis={exp_val:.4f}, "
                f"data={actual:.4f}"
            )

    if not warnings_log:
        print("  All thesis values confirmed.")


# ---------------------------------------------------------------------------
# Figure A — Pre-retrieval NDCG by representation (3-panel horizontal bar)
# ---------------------------------------------------------------------------

def make_fig_A(force: bool) -> None:
    df = pd.read_csv(SRC_PRE_ALL)
    df.columns = df.columns.str.strip()

    entity_order  = ["paper", "dataset", "model"]
    entity_labels = {
        "paper":   "Papers  (250 questions)",
        "dataset": "Datasets  (125 questions)",
        "model":   "Models  (125 questions)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))

    for ax, et in zip(axes, entity_order):
        group = df[df["entity_type"].str.strip() == et].copy()
        group["label"] = group["representation"].map(
            lambda r: REPR_DISPLAY.get(str(r).strip(), str(r).replace("_", " ").title())
        )
        group = group.sort_values("NDCG", ascending=True).reset_index(drop=True)
        best_ndcg = float(group["NDCG"].max())

        c_best  = ENTITY_COLORS[et]
        c_other = ENTITY_COLORS_LT[et]
        colors  = [c_best if abs(float(v) - best_ndcg) < 1e-9 else c_other
                   for v in group["NDCG"]]

        ax.barh(group["label"], group["NDCG"].astype(float),
                color=colors, height=0.55, edgecolor="white", linewidth=0.4)

        for idx, (_, row) in enumerate(group.iterrows()):
            val   = float(row["NDCG"])
            is_best = abs(val - best_ndcg) < 1e-9
            color = "white" if is_best else "#444444"
            ax.text(val - 0.006, idx, f"{val:.4f}",
                    va="center", ha="right", fontsize=FS - 2, color=color)

        ax.set_xlim(0, min(1.0, best_ndcg * 1.25 + 0.04))
        ax.set_xlabel("NDCG", fontsize=FS)
        ax.set_ylabel(entity_labels[et], fontsize=FS, labelpad=4)
        ax.tick_params(axis="y", labelsize=FS - 1)

    best_patch  = mpatches.Patch(color="#444444", label="Best strategy (highlighted per entity type)")
    other_patch = mpatches.Patch(color="#CCCCCC", label="Other strategies")
    fig.legend(handles=[best_patch, other_patch],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.002),
               fontsize=FS - 1, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "01_pre_retrieval_ndcg_by_representation", force)


# ---------------------------------------------------------------------------
# Figure 01b — Pre-retrieval NDCG by entity type × difficulty (3×3 heatmap)
# ---------------------------------------------------------------------------

BEST_REPR = {
    "paper":   "predicate_filtered",
    "dataset": "dataset_metadata",
    "model":   "model_metadata",
}

def make_fig_01b(force: bool) -> None:
    raw = json.load(open(SRC_PRE_DIFF, encoding="utf-8"))

    # Flatten all rows from all difficulty segments
    rows: list[dict] = []
    for diff, seg in raw.get("segments", {}).items():
        for r in seg.get("rows", []):
            rows.append(r)

    df = pd.DataFrame(rows)
    df.columns = df.columns.str.strip()
    df["entity_type"]    = df["entity_type"].str.strip()
    df["representation"] = df["representation"].str.strip()
    df["difficulty"]     = df["difficulty"].str.strip()

    # Keep only the best representation per entity type
    mask = pd.Series([
        BEST_REPR.get(str(et), "") == str(rep)
        for et, rep in zip(df["entity_type"], df["representation"])
    ])
    df = df[mask].copy()

    entity_order = ["paper", "dataset", "model"]
    entity_labels = {"paper": "Paper", "dataset": "Dataset", "model": "Model"}
    diff_order    = ["easy", "medium", "hard"]
    diff_labels   = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}

    # Build 3×3 NDCG and N arrays
    ndcg_data = np.full((3, 3), np.nan)
    n_data    = np.full((3, 3), 0, dtype=int)

    for ri, et in enumerate(entity_order):
        for ci, diff in enumerate(diff_order):
            sub = df[(df["entity_type"] == et) & (df["difficulty"] == diff)]
            if not sub.empty:
                ndcg_data[ri, ci] = float(sub.iloc[0]["NDCG"])
                n_data[ri, ci]    = int(sub.iloc[0]["evaluated_questions"])

    row_labels = [entity_labels[et] for et in entity_order]
    col_labels = [diff_labels[d]    for d in diff_order]

    vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    im = ax.imshow(ndcg_data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=FS)
    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels, fontsize=FS)
    ax.set_xlabel("Question Difficulty", fontsize=FS)
    ax.set_ylabel("Entity Type", fontsize=FS)

    for ri in range(3):
        for ci in range(3):
            val = ndcg_data[ri, ci]
            n   = n_data[ri, ci]
            if not np.isnan(val):
                txt_color = "white" if norm(val) > 0.60 else "black"
                ax.text(ci, ri - 0.12, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=FS, color=txt_color, fontweight="bold")
                ax.text(ci, ri + 0.22, f"N={n}",
                        ha="center", va="center",
                        fontsize=FS - 2, color=txt_color, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("NDCG", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS - 1)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout()
    save_figure(fig, "01b_pre_retrieval_ndcg_by_entity_difficulty", force)


# ---------------------------------------------------------------------------
# Figure B — Retrieval method comparison (all 6 methods; no-ops hatched)
# ---------------------------------------------------------------------------

def make_fig_B(force: bool) -> None:
    df = pd.read_csv(SRC_RET)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip()
    df["_order"] = df["method"].map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    short_labels = [METHOD_SHORT.get(m, m) for m in df["method"]]
    x         = np.arange(len(df))
    bar_width  = 0.32

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for i, (_, row) in enumerate(df.iterrows()):
        method   = str(row["method"])
        is_base  = method == "pure_semantic_dense"
        is_rrf   = "rrf" in method
        is_noop  = method in NOOP_METHODS
        hatch    = "///" if is_noop else None

        if is_base:
            c_ndcg = C_BASELINE; c_h10 = C_BASELINE
        elif is_rrf:
            c_ndcg = C_RRF;      c_h10 = C_RRF_HIT
        else:
            c_ndcg = C_HYBRID;   c_h10 = C_HIT

        ndcg_val = float(row["NDCG"])
        h10_val  = float(row["Hit@10"])

        ax.bar(x[i] - bar_width / 2, ndcg_val, bar_width,
               color=c_ndcg, alpha=0.88, edgecolor="white", linewidth=0.4,
               hatch=hatch)
        ax.bar(x[i] + bar_width / 2, h10_val, bar_width,
               color=c_h10, alpha=0.88, edgecolor="white", linewidth=0.4,
               hatch=hatch)

        # Numeric value labels above each bar
        label_kw = dict(ha="center", va="bottom", fontsize=FS - 3, color="#333333")
        ax.text(x[i] - bar_width / 2, ndcg_val + 0.004, f"{ndcg_val:.3f}", **label_kw)
        ax.text(x[i] + bar_width / 2, h10_val  + 0.004, f"{h10_val:.3f}",  **label_kw)

        # Annotate no-op methods
        if is_noop:
            top = max(ndcg_val, h10_val) + 0.012
            ax.text(x[i], top, NOOP_LABEL[method],
                    ha="center", va="bottom", fontsize=FS - 2,
                    color="#666666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=FS - 1)
    ax.set_ylabel("Score", fontsize=FS)
    ax.set_ylim(0, 0.82)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ndcg_patch    = mpatches.Patch(color=C_HYBRID,   label="NDCG (hybrid)")
    h10_patch     = mpatches.Patch(color=C_HIT,      label="Hit@10 (hybrid)")
    rrf_patch     = mpatches.Patch(color=C_RRF,      label="NDCG (RRF methods)")
    rrf_h10_patch = mpatches.Patch(color=C_RRF_HIT,  label="Hit@10 (RRF methods)")
    base_patch    = mpatches.Patch(color=C_BASELINE, label="Dense baseline")
    noop_patch    = mpatches.Patch(facecolor="#DDDDDD", hatch="///", edgecolor="#999999",
                                    label="Structural no-op (Hybrid-Type ≡ Dense)")
    ax.legend(handles=[base_patch, ndcg_patch, h10_patch,
                       rrf_patch, rrf_h10_patch, noop_patch],
              fontsize=FS - 1, frameon=False, ncol=3,
              loc="upper left", bbox_to_anchor=(0.01, 0.99))

    fig.tight_layout()
    save_figure(fig, "02_retrieval_method_comparison", force)


# ---------------------------------------------------------------------------
# Figure C — Question-type NDCG heatmap (5 distinct methods)
# ---------------------------------------------------------------------------

def make_fig_C(force: bool) -> None:
    df = pd.read_csv(SRC_QTYPE)
    df.columns = df.columns.str.strip()
    df["method"]        = df["method"].str.strip()
    df["question_type"] = df["question_type"].str.strip()

    max_counts   = df.groupby("question_type")["count"].max()
    valid_qtypes = max_counts[max_counts >= 3].index.tolist()

    dense_ndcg = (df[df["method"] == "pure_semantic_dense"]
                  .set_index("question_type")["NDCG"])
    valid_qtypes = sorted(valid_qtypes,
                          key=lambda q: -float(dense_ndcg.get(q, 0)))

    methods     = [m for m in DISTINCT_METHODS if m in df["method"].unique()]
    short_m     = [METHOD_SHORT[m] for m in methods]

    data = np.full((len(valid_qtypes), len(methods)), np.nan)
    for ri, qt in enumerate(valid_qtypes):
        for ci, m in enumerate(methods):
            sub = df[(df["question_type"] == qt) & (df["method"] == m)]
            if not sub.empty:
                data[ri, ci] = float(sub.iloc[0]["NDCG"])

    qt_labels = []
    for qt in valid_qtypes:
        n     = int(max_counts[qt])
        label = QTYPE_DISPLAY.get(qt, qt.replace("_", " ").title())
        qt_labels.append(f"{label}  (N={n})")

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    fig_h = max(5.5, len(valid_qtypes) * 0.5)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(short_m)))
    ax.set_xticklabels(short_m, fontsize=FS)
    ax.set_yticks(range(len(qt_labels)))
    ax.set_yticklabels(qt_labels, fontsize=FS - 1)
    ax.set_xlabel("Retrieval Method", fontsize=FS)
    ax.set_ylabel("Question Type", fontsize=FS)

    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val = data[ri, ci]
            if not np.isnan(val):
                nv        = norm(val)
                txt_color = "black" if 0.25 < nv < 0.80 else "white"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=FS - 1, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("NDCG", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS - 1)

    fig.tight_layout()
    save_figure(fig, "03_retrieval_question_type_heatmap", force)


# ---------------------------------------------------------------------------
# Figure D — Retrieval NDCG by entity type (5 distinct methods)
# ---------------------------------------------------------------------------

def make_fig_D(force: bool) -> None:
    df = pd.read_csv(SRC_ENTITY)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip()
    df = df[df["method"].isin(DISTINCT_METHODS)].copy()
    df["_order"] = df["method"].map(lambda m: DISTINCT_METHODS.index(m))
    df = df.sort_values("_order").reset_index(drop=True)

    cols   = ["paper_NDCG", "dataset_NDCG", "model_NDCG"]
    labels = ["Paper\n(250 q)", "Dataset\n(125 q)", "Model\n(125 q)"]
    data   = df[cols].values.astype(float)
    y_labs = [METHOD_SHORT[m] for m in df["method"]]

    vmin = float(data.min()) - 0.02
    vmax = float(data.max()) + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FS)
    ax.set_yticks(range(len(y_labs)))
    ax.set_yticklabels(y_labs, fontsize=FS)
    ax.set_xlabel("Entity Type", fontsize=FS)
    ax.set_ylabel("Retrieval Method", fontsize=FS)

    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val       = data[ri, ci]
            txt_color = "white" if norm(val) > 0.65 else "black"
            ax.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                    fontsize=FS - 1, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("NDCG", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS - 1)

    fig.tight_layout()
    save_figure(fig, "04_retrieval_ndcg_by_entity_type", force)


# ---------------------------------------------------------------------------
# Figure E — Retrieval NDCG by difficulty (5 distinct methods, grouped bars)
# ---------------------------------------------------------------------------

def make_fig_E(force: bool) -> None:
    df = pd.read_csv(SRC_DIFF)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip()
    df = df[df["method"].isin(DISTINCT_METHODS)].copy()
    df["_order"] = df["method"].map(lambda m: DISTINCT_METHODS.index(m))
    df = df.sort_values("_order").reset_index(drop=True)

    diff_cols   = ["easy_NDCG", "medium_NDCG", "hard_NDCG"]
    diff_labels = ["Easy\n(131 q)", "Medium\n(177 q)", "Hard\n(192 q)"]

    def _method_color(method: str) -> str:
        if method == "pure_semantic_dense":
            return C_BASELINE
        if "rrf_symbolic" in method:
            return C_RRF
        if "rrf" in method:
            return C_RRF_HIT
        if "predicate" in method:
            return C_HYBRID
        return C_HIT

    method_colors = [_method_color(str(row["method"])) for _, row in df.iterrows()]

    n     = len(df)
    bar_w = 0.15
    x     = np.arange(len(diff_cols))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for i, (_, row) in enumerate(df.iterrows()):
        vals   = [float(row[c]) for c in diff_cols]
        offset = (i - n / 2 + 0.5) * bar_w
        short  = METHOD_SHORT[str(row["method"])]
        ax.bar(x + offset, vals, bar_w, label=short,
               color=method_colors[i], alpha=0.88,
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels, fontsize=FS)
    ax.set_ylabel("NDCG", fontsize=FS)
    ax.set_ylim(0, 0.88)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS - 1, frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.22), loc="upper center")

    fig.tight_layout()
    save_figure(fig, "05_retrieval_ndcg_by_difficulty", force)


# ---------------------------------------------------------------------------
# Figure F — Hit@1 vs Hit@10 gap (all 6 methods; appendix)
# ---------------------------------------------------------------------------

def make_fig_F(force: bool) -> None:
    df = pd.read_csv(SRC_GAP)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip()
    df["_order"] = df["method"].map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99
    )
    df = df.sort_values("_order").reset_index(drop=True)

    short_labels = [METHOD_SHORT.get(m, m) for m in df["method"]]
    gap_col  = "gap_Hit10_minus_Hit1"
    bar_w    = 0.32
    x        = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for i, (_, row) in enumerate(df.iterrows()):
        method   = str(row["method"])
        is_base  = method == "pure_semantic_dense"
        is_rrf   = "rrf" in method
        is_noop  = method in NOOP_METHODS

        c_h1  = C_BASELINE if is_base else (C_RRF   if is_rrf else C_HYBRID)
        c_h10 = C_BASELINE if is_base else (C_RRF_HIT if is_rrf else C_HIT)
        hatch = "///" if is_noop else None

        ax.bar(x[i] - bar_w / 2, float(row["Hit@1"]),  bar_w,
               color=c_h1,  alpha=0.88, edgecolor="white", linewidth=0.4,
               hatch=hatch)
        ax.bar(x[i] + bar_w / 2, float(row["Hit@10"]), bar_w,
               color=c_h10, alpha=0.88, edgecolor="white", linewidth=0.4,
               hatch=hatch)

        gap = float(row[gap_col])
        top = max(float(row["Hit@1"]), float(row["Hit@10"])) + 0.016
        ax.annotate(f"+{gap:.3f}", xy=(x[i], top), ha="center", va="bottom",
                    fontsize=FS - 2, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=FS - 1)
    ax.set_ylabel("Score", fontsize=FS)
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    h1_patch   = mpatches.Patch(color=C_HYBRID,    label="Hit@1")
    h10_patch  = mpatches.Patch(color=C_HIT,       label="Hit@10")
    base_patch = mpatches.Patch(color=C_BASELINE,  label="Dense baseline")
    rrf_h1     = mpatches.Patch(color=C_RRF,       label="Hit@1 (RRF)")
    rrf_h10    = mpatches.Patch(color=C_RRF_HIT,   label="Hit@10 (RRF)")
    ax.legend(handles=[base_patch, h1_patch, h10_patch, rrf_h1, rrf_h10],
              fontsize=FS - 1, frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.22), loc="upper center")

    fig.tight_layout()
    save_figure(fig, "06_retrieval_hit_gap_optional", force)


# ---------------------------------------------------------------------------
# FIGURE_AUDIT.md
# ---------------------------------------------------------------------------

def write_audit_report() -> None:
    audit_path = OUT_DIR / "FIGURE_AUDIT.md"
    warn_block = "\n".join(f"- {w}" for w in warnings_log) if warnings_log else "None — all thesis values confirmed."

    content = dedent(f"""\
    # Figure Audit Report
    Generated by `scripts/generate_results_final_figures.py`

    ---

    ## A. Old figures found

    **docs/chapter4_artifacts/figures/**
    - fig_pre_retrieval_ndcg_by_representation.{{pdf,png}} — original, May 2026
    - fig_pre_retrieval_ndcg_by_representation_current.{{pdf,png}} — regenerated May 2026
    - fig_pre_retrieval_difficulty_ndcg_current.{{pdf,png}} — crowded multi-repr difficulty figure
    - fig_retrieval_method_comparison_current.{{pdf,png}} — does not mark no-op methods
    - fig_retrieval_hit_gap_current.{{pdf,png}}
    - fig_retrieval_question_type_heatmap_current.{{pdf,png}} — showed all 6 (redundant) methods
    - fig_retrieval_by_entity_type_current.{{pdf,png}} — showed all 6 (redundant) methods
    - fig_retrieval_by_difficulty_current.{{pdf,png}} — showed all 6 (redundant) methods

    **data/results/pre_retrieval/figures/**
    - overall_ndcg_by_representation.pdf — older combined plot
    - difficulty_ndcg_by_representation.pdf — older crowded difficulty plot

    ---

    ## B. Raw data files used

    | Figure | Source file |
    |---|---|
    | A (pre-retrieval NDCG) | data/results/pre_retrieval/summary_all_representations.csv |
    | 01b (entity × difficulty heatmap) | data/results/pre_retrieval/summary_by_difficulty.json |
    | B (retrieval comparison) | data/results/retrieval/summary.csv |
    | C (question-type heatmap) | data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv |
    | D (entity-type heatmap) | data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv |
    | E (difficulty bars) | data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv |
    | F (Hit gap) | data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv |

    ---

    ## C. Validation

    {warn_block}

    Checked values:
    - Paper best representation: predicate_filtered, NDCG 0.4544
    - Dataset best representation: dataset_metadata, NDCG 0.5281
    - Model best representation: model_metadata, NDCG 0.5046
    - Dense baseline: NDCG 0.4854, Hit@10 0.598
    - RRF-Fusion: NDCG 0.4962, Hit@10 0.618

    ---

    ## D. Final figure list

    | # | Filename | Purpose | Section | Placement | Replaces |
    |---|---|---|---|---|---|
    | A | 01_pre_retrieval_ndcg_by_representation | RQ1 main figure | §4.2 Pre-Retrieval | **Main chapter** | fig_pre_retrieval_ndcg_by_representation_current |
    | 01b | 01b_pre_retrieval_ndcg_by_entity_difficulty | Entity × difficulty NDCG heatmap | §4.2 Pre-Retrieval (difficulty subsection) | **Main chapter or Appendix** | fig_pre_retrieval_difficulty_ndcg_current (crowded, removed) |
    | B | 02_retrieval_method_comparison | RQ2 main figure | §4.3 Retrieval | **Main chapter** | fig_retrieval_method_comparison_current |
    | C | 03_retrieval_question_type_heatmap | Segmented analysis | §4.3 Retrieval | **Main chapter** | fig_retrieval_question_type_heatmap_current |
    | D | 04_retrieval_ndcg_by_entity_type | Entity breakdown | §4.3 Retrieval | Appendix | fig_retrieval_by_entity_type_current |
    | E | 05_retrieval_ndcg_by_difficulty | Difficulty breakdown | §4.3 Retrieval | **Main chapter** | fig_retrieval_by_difficulty_current |
    | F | 06_retrieval_hit_gap_optional | Hit@1 vs Hit@10 gap | Appendix | Appendix only | fig_retrieval_hit_gap_current |

    ---

    ## E. Recommendation

    **Main chapter (§4.2 and §4.3): use figures A, 01b (conditional), B, C, E**

    - **A** — the 3-panel horizontal bar is the clearest way to show all representation
      strategies per entity type and identify the best one for each. Supports RQ1 directly.

    - **B** — the 6-method comparison with hatching on the single no-op method (Hybrid-Type)
      and ≡-annotation directly communicates the structural redundancy finding: one method is
      identical to the Dense baseline because entity-type specificity is already enforced
      upstream by the collection design. The four remaining non-Dense methods (OneHop,
      Predicate, RRF-Fusion, RRF-Symbolic) are each behaviourally distinct and show a clear
      performance progression up to NDCG +0.024. This is the central RQ2 result figure.

    - **C** — the question-type heatmap (5 distinct methods) shows which question categories
      benefit from OneHop, Predicate, or RRF methods, and which remain weak. Essential for
      understanding where the pipeline succeeds and fails.

    - **E** — the difficulty bar chart (5 distinct methods) shows that easy questions benefit
      most from RRF while medium questions are the weakest segment across all methods.
      This supports the discussion of query difficulty as a systematic factor.

    - **01b** — the 3×3 entity-type × difficulty heatmap (best representation only) cleanly
      replaces the old crowded multi-representation difficulty bar chart. Use in the main
      chapter if §4.2 includes a Difficulty and Question-Type Analysis subsection; otherwise
      move to appendix.

    **Appendix only: D and F**

    - **D** — entity-type NDCG is a 3×3 heatmap that is better communicated as the
      existing thesis table. The figure adds little beyond what the numbers already say.
      Keep in appendix as a visual companion to the table.

    - **F** — the Hit@1 vs Hit@10 gap is already described clearly in the text and
      tables. The figure is correct but does not add a new argument. Appendix only.

    **Remove entirely:**
    - The old pre-retrieval difficulty-by-representation figure (fig_pre_retrieval_difficulty_ndcg_current)
      was crowded and showed all representations per difficulty level. Figure 01b replaces
      it with a clean 3×3 heatmap showing only the best representation per entity type,
      which is compact, readable, and argument-relevant.
    """)

    audit_path.write_text(content, encoding="utf-8")
    print(f"  [OK]    FIGURE_AUDIT.md")


# ---------------------------------------------------------------------------
# LATEX_SNIPPETS.tex
# ---------------------------------------------------------------------------

def write_latex_snippets() -> None:
    tex_path = OUT_DIR / "LATEX_SNIPPETS.tex"

    content = dedent(r"""
    % =========================================================================
    % LaTeX figure snippets for Chapter 4 final figures
    % Generated by scripts/generate_results_final_figures.py
    %
    % Recommended main chapter figures: A (01), 01b (conditional), B (02), C (03), E (05)
    % Appendix figures:                 D (04), F (06)
    %
    % Usage: place \graphicspath{{figs/results_final/}} in preamble, or use
    % full relative paths as shown below.
    % =========================================================================

    % -------------------------------------------------------------------------
    % Figure A — Pre-retrieval NDCG by representation  (§4.2)
    % -------------------------------------------------------------------------
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=\linewidth]{figs/results_final/01_pre_retrieval_ndcg_by_representation.pdf}
      \caption{%
        NDCG scores for all evaluated textual representation strategies, grouped by
        entity type. Each panel shows representations ordered by NDCG from lowest to
        highest; the highest-scoring strategy per entity type is highlighted.
        Best strategies: Predicate-Filtered for papers (NDCG\,=\,0.4544),
        Metadata for datasets (NDCG\,=\,0.5281), and Metadata for models
        (NDCG\,=\,0.5046). Evaluated on 500 answerable questions (250 paper,
        125 dataset, 125 model).
      }
      \label{fig:pre_retrieval_ndcg_by_representation}
    \end{figure}

    % -------------------------------------------------------------------------
    % Figure 01b — Pre-retrieval NDCG by entity type × difficulty  (§4.2)
    % Optional main-chapter figure — use if §4.2 contains a Difficulty and
    % Question-Type Analysis subsection; otherwise place in appendix.
    % -------------------------------------------------------------------------
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=0.68\linewidth]{figs/results_final/01b_pre_retrieval_ndcg_by_entity_difficulty.pdf}
      \caption{%
        NDCG of the selected best representation per entity type across question
        difficulty levels. Best representations: Predicate-Filtered (papers),
        Metadata (datasets and models). N indicates the number of evaluable questions
        per entity-type and difficulty combination. Easy questions consistently yield
        higher NDCG across all entity types; medium questions are the weakest
        segment, particularly for papers (NDCG\,=\,0.NNN).
      }
      \label{fig:pre_retrieval_ndcg_by_entity_difficulty}
    \end{figure}

    % -------------------------------------------------------------------------
    % Figure B — Retrieval method comparison  (§4.3)
    % -------------------------------------------------------------------------
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=\linewidth]{figs/results_final/02_retrieval_method_comparison.pdf}
      \caption{%
        NDCG and Hit@10 scores for all six retrieval methods evaluated on
        500 answerable questions.
        The hatched bar (\(\equiv\) annotation) indicates \texttt{hybrid\_type\_filtering},
        which is a structural no-op: the pre-retrieval stage already queries
        entity-type-specific collections, so type filtering removes no candidates.
        Of the five behaviourally distinct methods, Hybrid-OneHop yields a marginal
        gain (NDCG\,+0.0011), Hybrid-Predicate improves ordering within the
        top-10 list (NDCG\,+0.0053), RRF-Fusion achieves the largest top-10
        recall improvement (Hit@10\,+0.020, NDCG\,+0.011), and RRF-Symbolic
        achieves the strongest overall result (NDCG\,+0.024, Hit@1\,+0.018
        over the Dense baseline).
      }
      \label{fig:retrieval_method_comparison}
    \end{figure}

    % -------------------------------------------------------------------------
    % Figure C — Question-type NDCG heatmap  (§4.3)
    % -------------------------------------------------------------------------
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=\linewidth]{figs/results_final/03_retrieval_question_type_heatmap.pdf}
      \caption{%
        NDCG by question type for the five behaviourally distinct retrieval
        methods (Dense, OneHop, Predicate-Aware, RRF-Fusion, RRF-Symbolic).
        Rows are sorted by Dense baseline NDCG (descending); N gives the
        number of questions per type. High-confidence title-lookup questions
        (\emph{Title$\to$Entity}) reach NDCG $>$ 0.90, while low-resource
        multi-hop types (\emph{Task$\to$Paper}, \emph{Author$\to$Paper}) remain
        below 0.20 across all methods.
      }
      \label{fig:retrieval_question_type_heatmap}
    \end{figure}

    % -------------------------------------------------------------------------
    % Figure E — Retrieval NDCG by difficulty  (§4.3)
    % -------------------------------------------------------------------------
    \begin{figure}[htbp]
      \centering
      \includegraphics[width=0.85\linewidth]{figs/results_final/05_retrieval_ndcg_by_difficulty.pdf}
      \caption{%
        NDCG broken down by question difficulty for the five behaviourally
        distinct retrieval methods (Dense, OneHop, Predicate-Aware, RRF-Fusion,
        RRF-Symbolic). Easy questions benefit most from RRF-Fusion
        (NDCG\,=\,0.6904 vs.\ 0.6652 for Dense). Medium questions remain the
        weakest segment (NDCG\,$\leq$\,0.34) across all methods, indicating that
        multi-constraint queries present a persistent retrieval challenge
        independent of the ranking strategy.
      }
      \label{fig:retrieval_ndcg_by_difficulty}
    \end{figure}

    % =========================================================================
    % Appendix figures (D and F) — include below if needed
    % =========================================================================

    % Figure D — Entity-type NDCG heatmap  (Appendix)
    % \begin{figure}[htbp]
    %   \centering
    %   \includegraphics[width=0.65\linewidth]{figs/results_final/04_retrieval_ndcg_by_entity_type.pdf}
    %   \caption{%
    %     NDCG by entity type for the three distinct retrieval methods.
    %     Dataset retrieval benefits least from RRF-Fusion, while model retrieval
    %     shows the largest gain (+0.029 NDCG over Dense).
    %   }
    %   \label{fig:retrieval_ndcg_by_entity_type}
    % \end{figure}

    % Figure F — Hit@1 vs Hit@10 gap  (Appendix)
    % \begin{figure}[htbp]
    %   \centering
    %   \includegraphics[width=\linewidth]{figs/results_final/06_retrieval_hit_gap_optional.pdf}
    %   \caption{%
    %     Hit@1 and Hit@10 for all six retrieval methods, with the recoverability
    %     gap (Hit@10$-$Hit@1) annotated above each bar pair.
    %     RRF-Fusion improves top-10 coverage (Hit@10\,=\,0.618, gap\,=\,+0.240),
    %     indicating that multi-representation fusion brings new gold entities into
    %     the top-10 pool. RRF-Symbolic additionally improves top-1 ranking
    %     (Hit@1\,=\,0.396), narrowing the gap to +0.222 while maintaining the
    %     same top-10 coverage as RRF-Fusion.
    %   }
    %   \label{fig:retrieval_hit_gap}
    % \end{figure}
    """).lstrip()

    tex_path.write_text(content, encoding="utf-8")
    print(f"  [OK]    LATEX_SNIPPETS.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final Chapter 4 thesis figures."
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing figures.")
    args = parser.parse_args()

    print("=" * 62)
    print("Chapter 4 Final Figure Generator")
    print("=" * 62)

    print("\nValidating source files and thesis values ...")
    validate()

    print("\nGenerating figures ...")
    make_fig_A(args.force)
    make_fig_01b(args.force)
    make_fig_B(args.force)
    make_fig_C(args.force)
    make_fig_D(args.force)
    make_fig_E(args.force)
    make_fig_F(args.force)

    print("\nWriting audit report and LaTeX snippets ...")
    write_audit_report()
    write_latex_snippets()

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    try:
        print(f"\nOutput directory : {OUT_DIR}")
    except UnicodeEncodeError:
        print(f"\nOutput directory : figs/results_final/")
    print(f"Figures generated: {len(generated)}")

    if warnings_log:
        print(f"\nValidation warnings ({len(warnings_log)}):")
        for w in warnings_log:
            print(f"  ! {w}")
    else:
        print("\nValidation        : all thesis values confirmed, no warnings")

    print("\nMain chapter figures   : A (01), 01b (conditional), B (02), C (03), E (05)")
    print("Appendix figures       : D (04), F (06)")
    print("Old difficulty figure  : remove (replaced by 01b)")

    if skipped_list:
        print(f"\nSkipped ({len(skipped_list)}) — already exist (use --force):")
        for name, _ in skipped_list:
            print(f"  {name}")


if __name__ == "__main__":
    main()
