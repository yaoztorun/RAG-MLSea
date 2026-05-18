#!/usr/bin/env python3
"""
generate_chapter4_artifacts.py

Generates thesis-ready Chapter 4 LaTeX tables and figures for the
pre-retrieval and retrieval evaluation stages of the MLSea KG-RAG pipeline.

Run from repo root:
    python docs/chapter4_artifacts/scripts/generate_chapter4_artifacts.py

Requires: pandas, matplotlib, numpy
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]          # scripts/ -> chapter4_artifacts/ -> docs/ -> root

SRC = {
    "questions":       REPO_ROOT / "data" / "questions" / "ml_questions_dataset.json",
    "best_entity":     REPO_ROOT / "data" / "results" / "thesis_tables" / "best_per_entity.csv",
    "full_comp":       REPO_ROOT / "data" / "results" / "thesis_tables" / "full_comparison.csv",
    "difficulty":      REPO_ROOT / "data" / "results" / "thesis_tables" / "difficulty_breakdown.csv",
    "ret_main":        REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_main_comparison.csv",
    "ret_entity":      REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_entity_type_ndcg.csv",
    "ret_diff":        REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_difficulty_ndcg.csv",
    "ret_qtype":       REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_question_type_ndcg.csv",
    "ret_gap":         REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_precision_recall_tradeoff.csv",
    "diff_json":       REPO_ROOT / "data" / "results" / "retrieval" / "summary_by_difficulty.json",
    "thesis_figs":     REPO_ROOT / "data" / "results" / "thesis_figures",
}

OUT_TABLES  = REPO_ROOT / "docs" / "chapter4_artifacts" / "tables"
OUT_FIGURES = REPO_ROOT / "docs" / "chapter4_artifacts" / "figures"

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

ENTITY_COLORS = {
    "paper":   "#4472C4",
    "dataset": "#E07B54",
    "model":   "#70A45E",
}
METHOD_COLORS = [
    "#4472C4", "#70A45E", "#E07B54", "#B07AA1", "#F28E2B", "#76B7B2"
]
BEST_COLOR    = "#1A3A6B"
OTHER_COLOR   = "#A8BDD8"
BASELINE_COLOR = "#AAAAAA"

METHOD_SHORT = {
    "pure_semantic_dense":               "Dense",
    "hybrid_type_filtering":             "Hybrid-Type",
    "hybrid_type_onehop_filtering":      "Hybrid-OneHop",
    "hybrid_predicate_aware_filtering":  "Hybrid-Pred.",
    "optional_rrf_fusion":               "RRF-Fusion",
    "optional_rrf_symbolic":             "RRF-Symbolic",
}

METHOD_DISPLAY = {
    "pure_semantic_dense":               "Dense (baseline)",
    "hybrid_type_filtering":             "Hybrid: Type Filtering",
    "hybrid_type_onehop_filtering":      "Hybrid: Type + One-Hop",
    "hybrid_predicate_aware_filtering":  "Hybrid: Predicate-Aware",
    "optional_rrf_fusion":               "RRF: Multi-Repr. Fusion",
    "optional_rrf_symbolic":             "RRF: Fusion + Symbolic",
}

METHOD_GROUP = {
    "pure_semantic_dense":               "Pure Semantic",
    "hybrid_type_filtering":             "Hybrid",
    "hybrid_type_onehop_filtering":      "Hybrid",
    "hybrid_predicate_aware_filtering":  "Hybrid",
    "optional_rrf_fusion":               "RRF Fusion",
    "optional_rrf_symbolic":             "RRF Fusion",
}

FONT_SIZE = 9
DPI       = 300

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FONT_SIZE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    FONT_SIZE,
    "axes.labelsize":    FONT_SIZE,
    "xtick.labelsize":   FONT_SIZE - 1,
    "ytick.labelsize":   FONT_SIZE - 1,
    "legend.fontsize":   FONT_SIZE - 1,
})

# question-type → entity type mapping
PAPER_QTYPES = {
    "paper_by_author_and_task", "paper_by_task_pair", "paper_description_to_title",
    "paper_to_author_count", "paper_to_authors", "paper_to_implementation",
    "paper_to_keywords", "paper_to_publication_year", "paper_to_task_count", "paper_to_tasks",
}
DATASET_QTYPES = {
    "dataset_to_publication_year", "dataset_to_task_count", "dataset_to_task_membership",
    "dataset_to_tasks", "tasks_to_dataset", "semantic_task_to_dataset",
}
MODEL_QTYPES = {
    "model_family_variant", "model_name_resolution", "comparative_model_variant",
    "repository_to_model", "semantic_repository_to_model",
}

QTYPE_READABLE = {
    "paper_to_author_count":       "Paper: Author Count",
    "paper_to_tasks":              "Paper: Tasks",
    "paper_to_authors":            "Paper: Authors",
    "paper_to_implementation":     "Paper: Implementation",
    "paper_to_publication_year":   "Paper: Pub.~Year",
    "paper_to_task_count":         "Paper: Task Count",
    "paper_to_keywords":           "Paper: Keywords",
    "paper_description_to_title":  "Paper: Desc.\\,$\\to$\\,Title",
    "paper_by_task_pair":          "Paper: by Task Pair",
    "paper_by_author_and_task":    "Paper: by Author+Task",
    "repository_to_model":         "Model: Repo\\,$\\to$\\,Model",
    "comparative_model_variant":   "Model: Variant",
    "model_family_variant":        "Model: Family Variant",
    "dataset_to_tasks":            "Dataset: Tasks",
    "dataset_to_task_count":       "Dataset: Task Count",
    "dataset_to_task_membership":  "Dataset: Task Membership",
    "tasks_to_dataset":            "Tasks\\,$\\to$\\,Dataset",
    "semantic_task_to_dataset":    "Semantic Task\\,$\\to$\\,Dataset",
    # filtered out (N < 3):
    "model_name_resolution":       "Model: Name Resolution",
    "dataset_to_publication_year": "Dataset: Pub.~Year",
    "semantic_repository_to_model":"Semantic Repo\\,$\\to$\\,Model",
}

# plain text version for matplotlib labels (no LaTeX)
QTYPE_READABLE_PLAIN = {
    "paper_to_author_count":       "Paper: Author Count",
    "paper_to_tasks":              "Paper: Tasks",
    "paper_to_authors":            "Paper: Authors",
    "paper_to_implementation":     "Paper: Implementation",
    "paper_to_publication_year":   "Paper: Pub. Year",
    "paper_to_task_count":         "Paper: Task Count",
    "paper_to_keywords":           "Paper: Keywords",
    "paper_description_to_title":  "Paper: Desc.→Title",
    "paper_by_task_pair":          "Paper: by Task Pair",
    "paper_by_author_and_task":    "Paper: by Author+Task",
    "repository_to_model":         "Model: Repo→Model",
    "comparative_model_variant":   "Model: Variant",
    "model_family_variant":        "Model: Family Variant",
    "dataset_to_tasks":            "Dataset: Tasks",
    "dataset_to_task_count":       "Dataset: Task Count",
    "dataset_to_task_membership":  "Dataset: Task Membership",
    "tasks_to_dataset":            "Tasks→Dataset",
    "semantic_task_to_dataset":    "Semantic Task→Dataset",
    "model_name_resolution":       "Model: Name Resolution",
    "dataset_to_publication_year": "Dataset: Pub. Year",
    "semantic_repository_to_model":"Semantic Repo→Model",
}

generated_tables  = []
generated_figures = []
warnings_list     = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_inputs():
    missing = [k for k, p in SRC.items() if k != "thesis_figs" and not p.exists()]
    if missing:
        print("ERROR: Missing required source files:")
        for k in missing:
            print(f"  [{k}] {SRC[k]}")
        sys.exit(1)

def fmt(val, digits=4):
    """Format a numeric value to fixed decimal places."""
    if pd.isna(val):
        return "--"
    return f"{float(val):.{digits}f}"

def fmt_delta(val, digits=4):
    """Format delta with explicit + sign."""
    if pd.isna(val):
        return "--"
    v = float(val)
    if abs(v) < 1e-9:
        return f"0.{('0'*digits)}"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{digits}f}"

def escape_latex(s):
    """Escape common LaTeX special characters in a string."""
    return (str(s)
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("$", r"\$"))

def clean_repr_name(name, entity_type=None):
    """Strip entity prefix from representation name and title-case."""
    name = str(name)
    if entity_type:
        prefix = entity_type + "_"
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ").title()

def save_table(tex, filename):
    path = OUT_TABLES / filename
    path.write_text(tex, encoding="utf-8")
    generated_tables.append(filename)
    print(f"  [TABLE] {filename}")

def save_figure(fig, stem):
    pdf_path = OUT_FIGURES / f"{stem}.pdf"
    png_path = OUT_FIGURES / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    generated_figures.append(f"{stem}.pdf")
    generated_figures.append(f"{stem}.png")
    print(f"  [FIGURE] {stem}.pdf / .png")

def bold_max_in_col(col_values):
    """Return list of LaTeX strings with \textbf{} on max value."""
    nums = [float(v) if v != "--" else None for v in col_values]
    valid = [v for v in nums if v is not None]
    if not valid:
        return col_values
    max_val = max(valid)
    result = []
    for raw, num in zip(col_values, nums):
        if num is not None and abs(num - max_val) < 1e-9:
            result.append(r"\textbf{" + raw + r"}")
        else:
            result.append(raw)
    return result


# ---------------------------------------------------------------------------
# TABLE A — Evaluation Dataset Summary
# ---------------------------------------------------------------------------

def make_table_a():
    with open(SRC["questions"], encoding="utf-8") as f:
        questions = json.load(f)

    answerable   = [q for q in questions if q.get("is_answerable", True)]
    unanswerable = [q for q in questions if not q.get("is_answerable", True)]

    n_paper   = sum(1 for q in answerable if q.get("question_type", "") in PAPER_QTYPES)
    n_dataset = sum(1 for q in answerable if q.get("question_type", "") in DATASET_QTYPES)
    n_model   = sum(1 for q in answerable if q.get("question_type", "") in MODEL_QTYPES)
    n_qtypes  = len({q.get("question_type") for q in answerable})

    with open(SRC["diff_json"], encoding="utf-8") as f:
        diff_json = json.load(f)

    diff_counts = diff_json.get("pure_semantic_dense", {})
    n_easy    = diff_counts.get("easy",    {}).get("count", "?")
    n_medium  = diff_counts.get("medium",  {}).get("count", "?")
    n_hard    = diff_counts.get("hard",    {}).get("count", "?")
    n_unknown = diff_counts.get("unknown", {}).get("count", "?")

    rows = [
        ("Total questions",                          len(questions)),
        ("Answerable (used in evaluation)",          len(answerable)),
        ("Unanswerable (excluded)",                  len(unanswerable)),
        ("_sep1",                                    None),
        ("Paper questions",                          n_paper),
        ("Dataset questions",                        n_dataset),
        ("Model questions",                          n_model),
        ("_sep2",                                    None),
        ("Distinct question types",                  n_qtypes),
        ("_head",                                    None),
        ("\\quad Easy",                              n_easy),
        ("\\quad Medium",                            n_medium),
        ("\\quad Hard",                              n_hard),
        ("\\quad Unknown",                           n_unknown),
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Evaluation dataset summary. Difficulty counts apply to the "
                 r"answerable subset used in quantitative evaluation.}")
    lines.append(r"  \label{tab:evaluation_dataset_summary}")
    lines.append(r"  \begin{tabular}{lc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Property} & \textbf{Count} \\")
    lines.append(r"    \midrule")

    for label, val in rows:
        if label == "_sep1" or label == "_sep2":
            lines.append(r"    \addlinespace")
            continue
        if label == "_head":
            lines.append(r"    \addlinespace")
            lines.append(r"    \multicolumn{2}{l}{\textit{Difficulty (answerable questions)}} \\")
            continue
        lines.append(f"    {label} & {val} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_evaluation_dataset_summary.tex")


# ---------------------------------------------------------------------------
# TABLE B — Best Representation per Entity Type
# ---------------------------------------------------------------------------

def make_table_b():
    df = pd.read_csv(SRC["best_entity"])
    df.columns = df.columns.str.strip()

    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Best entity-centric representation strategy per entity type, "
                 r"evaluated at top-10 candidate entities.}")
    lines.append(r"  \label{tab:best_representation_by_entity}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Entity Type} & \textbf{Representation} & "
                 r"\textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{MRR} & \textbf{NDCG} \\")
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        et   = str(row["entity_type"]).strip().title()
        rep  = clean_repr_name(str(row["best_representation"]).strip(), str(row["entity_type"]).strip())
        vals = " & ".join(fmt(row[m]) for m in metrics)
        lines.append(f"    {et} & {rep} & {vals} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_best_representation_by_entity.tex")


# ---------------------------------------------------------------------------
# TABLE C — Full Pre-Retrieval Representation Comparison
# ---------------------------------------------------------------------------

def make_table_c():
    df = pd.read_csv(SRC["full_comp"])
    df.columns = df.columns.str.strip()

    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]
    entity_order = ["paper", "dataset", "model"]
    entity_sizes = {"paper": 6, "dataset": 4, "model": 4}

    lines = []
    lines.append(r"% Requires \usepackage{booktabs} and \usepackage{multirow}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Pre-retrieval representation comparison across all 14 strategies "
                 r"and three entity types. NDCG values are computed over top-10 candidate entities. "
                 r"Bold denotes the highest NDCG within each entity type.}")
    lines.append(r"  \label{tab:pre_retrieval_full_comparison}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Entity Type} & \textbf{Representation} & "
                 r"\textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{MRR} & \textbf{NDCG} \\")
    lines.append(r"    \midrule")

    first_entity = True
    for et in entity_order:
        group = df[df["entity_type"].str.strip() == et].copy()
        group = group.sort_values("NDCG", ascending=False).reset_index(drop=True)

        if not first_entity:
            lines.append(r"    \midrule")
        first_entity = False

        best_ndcg = group["NDCG"].max()
        n_rows = len(group)

        for i, (_, row) in enumerate(group.iterrows()):
            rep = clean_repr_name(str(row["representation"]).strip(), et)
            ndcg_val = float(row["NDCG"])
            ndcg_str = fmt(ndcg_val)
            if abs(ndcg_val - best_ndcg) < 1e-9:
                ndcg_str = r"\textbf{" + ndcg_str + r"}"

            metric_strs = [fmt(row[m]) for m in metrics[:-1]] + [ndcg_str]
            vals = " & ".join(metric_strs)

            if i == 0:
                et_cell = r"\multirow{" + str(n_rows) + r"}{*}{" + et.title() + r"}"
            else:
                et_cell = ""

            lines.append(f"    {et_cell} & {rep} & {vals} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_pre_retrieval_full_comparison.tex")


# ---------------------------------------------------------------------------
# TABLE D — Pre-Retrieval Difficulty Breakdown (best representations only)
# ---------------------------------------------------------------------------

def make_table_d():
    df = pd.read_csv(SRC["difficulty"])
    df.columns = df.columns.str.strip()

    best_repr = {
        "paper":   "enriched_metadata",
        "dataset": "dataset_title_only",
        "model":   "model_predicate_filtered",
    }

    diff_order = ["easy", "medium", "hard", "unknown"]
    entity_order = ["paper", "dataset", "model"]

    lines = []
    lines.append(r"% Requires \usepackage{booktabs} and \usepackage{multirow}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Difficulty breakdown for the best-performing representation "
                 r"per entity type. Metrics computed over top-10 candidate entities.}")
    lines.append(r"  \label{tab:pre_retrieval_difficulty}")
    lines.append(r"  \begin{tabular}{llcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Entity Type} & \textbf{Difficulty} & "
                 r"\textbf{Representation} & \textbf{Hit@1} & \textbf{MRR} & \textbf{NDCG} \\")
    lines.append(r"    \midrule")

    first_entity = True
    for et in entity_order:
        rep_name = best_repr[et]
        sub = df[
            (df["entity_type"].str.strip() == et) &
            (df["representation"].str.strip() == rep_name)
        ].copy()

        # Order by difficulty
        sub["_ord"] = sub["difficulty"].str.strip().apply(
            lambda d: diff_order.index(d) if d in diff_order else 99
        )
        sub = sub.sort_values("_ord").reset_index(drop=True)

        if sub.empty:
            warnings_list.append(f"TABLE D: No rows for entity_type={et}, representation={rep_name}")
            continue

        if not first_entity:
            lines.append(r"    \addlinespace")
        first_entity = False

        rep_readable = clean_repr_name(rep_name, et)
        n_rows = len(sub)

        for i, (_, row) in enumerate(sub.iterrows()):
            diff_label = str(row["difficulty"]).strip().title()
            if i == 0:
                et_cell = r"\multirow{" + str(n_rows) + r"}{*}{" + et.title() + r"}"
            else:
                et_cell = ""

            lines.append(
                f"    {et_cell} & {diff_label} & {rep_readable} & "
                f"{fmt(row['Hit@1'])} & {fmt(row['MRR'])} & {fmt(row['NDCG'])} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_pre_retrieval_difficulty.tex")


# ---------------------------------------------------------------------------
# TABLE E — Overall Retrieval Method Comparison
# ---------------------------------------------------------------------------

def make_table_e():
    df = pd.read_csv(SRC["ret_main"])
    df.columns = df.columns.str.strip()

    best_ndcg = df["NDCG"].max()
    baseline_method = "pure_semantic_dense"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Retrieval method comparison across six methods evaluated "
                 r"on 265 questions over top-10 candidate entities. "
                 r"$\Delta$NDCG is relative to the dense semantic baseline. "
                 r"\textsuperscript{\dag}~Dense semantic baseline.}")
    lines.append(r"  \label{tab:retrieval_method_comparison}")
    lines.append(r"  \begin{tabular}{llcccccr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{Group} & "
                 r"\textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & "
                 r"\textbf{MRR} & \textbf{NDCG} & \textbf{$\Delta$NDCG} \\")
    lines.append(r"    \midrule")

    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]

    for _, row in df.iterrows():
        method   = str(row["method"]).strip()
        display  = METHOD_DISPLAY.get(method, method)
        group    = METHOD_GROUP.get(method, "")
        delta    = fmt_delta(row.get("delta_NDCG_vs_dense", 0))
        ndcg_val = float(row["NDCG"])
        ndcg_str = fmt(ndcg_val)
        if abs(ndcg_val - best_ndcg) < 1e-9:
            ndcg_str = r"\textbf{" + ndcg_str + r"}"

        metric_strs = [fmt(row[m]) for m in metrics[:-1]] + [ndcg_str]
        vals = " & ".join(metric_strs)

        if method == baseline_method:
            display = display + r"\textsuperscript{\dag}"

        lines.append(f"    {escape_latex(display)} & {escape_latex(group)} & {vals} & {delta} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_retrieval_method_comparison.tex")


# ---------------------------------------------------------------------------
# TABLE F — Retrieval NDCG by Entity Type
# ---------------------------------------------------------------------------

def make_table_f():
    df = pd.read_csv(SRC["ret_entity"])
    df.columns = df.columns.str.strip()

    paper_col   = "paper_NDCG"
    dataset_col = "dataset_NDCG"
    model_col   = "model_NDCG"

    best_paper   = df[paper_col].max()
    best_dataset = df[dataset_col].max()
    best_model   = df[model_col].max()

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Retrieval NDCG broken down by entity type. "
                 r"Bold denotes the highest NDCG per column.}")
    lines.append(r"  \label{tab:retrieval_by_entity_type}")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{Paper} & \textbf{Dataset} & \textbf{Model} \\")
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        method  = str(row["method"]).strip()
        display = METHOD_DISPLAY.get(method, method)

        p_str = fmt(row[paper_col])
        d_str = fmt(row[dataset_col])
        m_str = fmt(row[model_col])

        if abs(float(row[paper_col]) - best_paper) < 1e-9:
            p_str = r"\textbf{" + p_str + r"}"
        if abs(float(row[dataset_col]) - best_dataset) < 1e-9:
            d_str = r"\textbf{" + d_str + r"}"
        if abs(float(row[model_col]) - best_model) < 1e-9:
            m_str = r"\textbf{" + m_str + r"}"

        lines.append(f"    {escape_latex(display)} & {p_str} & {d_str} & {m_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_retrieval_by_entity_type.tex")


# ---------------------------------------------------------------------------
# TABLE G — Retrieval NDCG by Difficulty
# ---------------------------------------------------------------------------

def make_table_g():
    df = pd.read_csv(SRC["ret_diff"])
    df.columns = df.columns.str.strip()

    cols = ["easy_NDCG", "medium_NDCG", "hard_NDCG", "unknown_NDCG"]
    best = {c: df[c].max() for c in cols}

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Retrieval NDCG broken down by question difficulty. "
                 r"Bold denotes the highest NDCG per column.}")
    lines.append(r"  \label{tab:retrieval_by_difficulty}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{Easy} & \textbf{Medium} "
                 r"& \textbf{Hard} & \textbf{Unknown} \\")
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        method  = str(row["method"]).strip()
        display = METHOD_DISPLAY.get(method, method)

        cells = []
        for c in cols:
            s = fmt(row[c])
            if abs(float(row[c]) - best[c]) < 1e-9:
                s = r"\textbf{" + s + r"}"
            cells.append(s)

        lines.append(f"    {escape_latex(display)} & {' & '.join(cells)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_retrieval_by_difficulty.tex")


# ---------------------------------------------------------------------------
# TABLE H — Hit@1 vs Hit@10 Recoverability Gap
# ---------------------------------------------------------------------------

def make_table_h():
    df = pd.read_csv(SRC["ret_gap"])
    df.columns = df.columns.str.strip()

    gap_col = "gap_Hit10_minus_Hit1"
    best_gap = df[gap_col].max()

    lines = []
    lines.append(r"% Gap = Hit@10 - Hit@1; measures how many correct entities are")
    lines.append(r"% recoverable in the top-10 but not ranked first.")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Recoverability gap between Hit@1 and Hit@10 across retrieval methods. "
                 r"A larger gap indicates more correct candidate entities present in the top-10 "
                 r"but not ranked first. Bold denotes the largest gap.}")
    lines.append(r"  \label{tab:retrieval_hit_gap}")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{Hit@1} & \textbf{Hit@10} "
                 r"& \textbf{Gap (Hit@10\,$-$\,Hit@1)} \\")
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        method  = str(row["method"]).strip()
        display = METHOD_DISPLAY.get(method, method)

        h1  = fmt(row["Hit@1"])
        h10 = fmt(row["Hit@10"])
        gap = fmt(row[gap_col])

        if abs(float(row[gap_col]) - best_gap) < 1e-9:
            gap = r"\textbf{" + gap + r"}"

        lines.append(f"    {escape_latex(display)} & {h1} & {h10} & {gap} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_retrieval_hit_gap.tex")


# ---------------------------------------------------------------------------
# TABLE I — Question-Type Retrieval Summary (compact)
# ---------------------------------------------------------------------------

def make_table_i():
    df = pd.read_csv(SRC["ret_qtype"])
    df.columns = df.columns.str.strip()

    selected_types = [
        "paper_to_author_count",
        "paper_to_tasks",
        "paper_to_implementation",
        "paper_description_to_title",
        "dataset_to_task_membership",
        "dataset_to_tasks",
        "paper_by_author_and_task",
        "semantic_task_to_dataset",
        "tasks_to_dataset",
    ]

    dense_method = "pure_semantic_dense"
    rrf_method   = "optional_rrf_symbolic"

    def get_row(qtype, method):
        sub = df[(df["question_type"].str.strip() == qtype) &
                 (df["method"].str.strip() == method)]
        if sub.empty:
            return None
        return sub.iloc[0]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Question-type retrieval summary: selected question types "
                 r"representing best-performing, worst-performing, and RRF-improved categories. "
                 r"$N$ = question count; NDCG computed over top-10 candidate entities. "
                 r"$\Delta$ = RRF$+$Symbolic\,$-$\,Dense. "
                 r"Bold denotes the higher NDCG between the two reported methods per row.}")
    lines.append(r"  \label{tab:retrieval_question_type_summary}")
    lines.append(r"  \begin{tabular}{lrccrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Question Type} & $N$ & "
                 r"\textbf{Dense NDCG} & \textbf{RRF+Sym. NDCG} & $\Delta$ \\")
    lines.append(r"    \midrule")

    for qt in selected_types:
        r_dense = get_row(qt, dense_method)
        r_rrf   = get_row(qt, rrf_method)

        if r_dense is None and r_rrf is None:
            warnings_list.append(f"TABLE I: No data for question_type={qt}")
            continue

        n = int(r_dense["count"]) if r_dense is not None else int(r_rrf["count"])
        dense_ndcg = float(r_dense["NDCG"]) if r_dense is not None else float("nan")
        rrf_ndcg   = float(r_rrf["NDCG"])   if r_rrf   is not None else float("nan")

        if not (np.isnan(dense_ndcg) or np.isnan(rrf_ndcg)):
            delta = rrf_ndcg - dense_ndcg
            delta_str = fmt_delta(delta)
            if dense_ndcg >= rrf_ndcg:
                d_str = r"\textbf{" + fmt(dense_ndcg) + r"}"
                r_str = fmt(rrf_ndcg)
            else:
                d_str = fmt(dense_ndcg)
                r_str = r"\textbf{" + fmt(rrf_ndcg) + r"}"
        else:
            delta_str = "--"
            d_str = fmt(dense_ndcg) if not np.isnan(dense_ndcg) else "--"
            r_str = fmt(rrf_ndcg)   if not np.isnan(rrf_ndcg)   else "--"

        label = QTYPE_READABLE.get(qt, qt.replace("_", " ").title())

        lines.append(f"    {label} & {n} & {d_str} & {r_str} & {delta_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_retrieval_question_type_summary.tex")


# ---------------------------------------------------------------------------
# FIGURE A — Pre-Retrieval NDCG by Representation Strategy (3-panel)
# ---------------------------------------------------------------------------

def make_figure_a():
    df = pd.read_csv(SRC["full_comp"])
    df.columns = df.columns.str.strip()

    entity_order  = ["paper", "dataset", "model"]
    entity_labels = {"paper": "Papers", "dataset": "Datasets", "model": "Models"}

    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    for ax, et in zip(axes, entity_order):
        group = df[df["entity_type"].str.strip() == et].copy()
        group["rep_label"] = group["representation"].apply(
            lambda r: clean_repr_name(r, et)
        )
        group = group.sort_values("NDCG", ascending=True).reset_index(drop=True)

        best_ndcg = group["NDCG"].max()
        colors = [BEST_COLOR if abs(v - best_ndcg) < 1e-9 else OTHER_COLOR
                  for v in group["NDCG"]]

        bars = ax.barh(group["rep_label"], group["NDCG"], color=colors, height=0.6)

        # Annotate best bar
        best_idx = group["NDCG"].idxmax()
        best_val = group.loc[best_idx, "NDCG"]
        ax.annotate(
            f"  {best_val:.4f}",
            xy=(best_val, best_idx),
            va="center", ha="left",
            fontsize=FONT_SIZE - 1,
            color=BEST_COLOR,
        )

        xlim = 0.55 if et == "dataset" else 1.0
        ax.set_xlim(0, xlim)
        ax.set_xlabel("NDCG")
        ax.set_ylabel(entity_labels[et], fontsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    best_patch  = mpatches.Patch(color=BEST_COLOR,  label="Best strategy")
    other_patch = mpatches.Patch(color=OTHER_COLOR, label="Other strategies")
    fig.legend(handles=[best_patch, other_patch],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
               fontsize=FONT_SIZE - 1, frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "fig_pre_retrieval_ndcg_by_representation")


# ---------------------------------------------------------------------------
# FIGURE B — Best Representation Performance by Difficulty
# ---------------------------------------------------------------------------

def make_figure_b():
    df = pd.read_csv(SRC["difficulty"])
    df.columns = df.columns.str.strip()

    best_repr = {
        "paper":   "enriched_metadata",
        "dataset": "dataset_title_only",
        "model":   "model_predicate_filtered",
    }

    diff_order   = ["easy", "medium", "hard"]
    entity_order = ["paper", "dataset", "model"]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    n_entities = len(entity_order)
    n_diffs    = len(diff_order)
    bar_width  = 0.22
    x          = np.arange(n_diffs)

    for i, et in enumerate(entity_order):
        rep = best_repr[et]
        sub = df[
            (df["entity_type"].str.strip() == et) &
            (df["representation"].str.strip() == rep)
        ]

        vals = []
        for d in diff_order:
            row = sub[sub["difficulty"].str.strip() == d]
            if row.empty:
                vals.append(np.nan)
                warnings_list.append(f"FIGURE B: No data for entity={et}, difficulty={d}")
            else:
                vals.append(float(row.iloc[0]["NDCG"]))

        offset = (i - n_entities / 2 + 0.5) * bar_width
        label  = clean_repr_name(rep, et)
        color  = ENTITY_COLORS[et]

        for j, (xval, yval) in enumerate(zip(x + offset, vals)):
            if not np.isnan(yval):
                ax.bar(xval, yval, width=bar_width, color=color,
                       label=et.title() if j == 0 else "_nolegend_",
                       alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("NDCG")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Entity Type", fontsize=FONT_SIZE - 1, title_fontsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, "fig_pre_retrieval_difficulty_ndcg")


# ---------------------------------------------------------------------------
# FIGURE C — Retrieval Method Comparison (NDCG + Hit@1)
# ---------------------------------------------------------------------------

def make_figure_c():
    df = pd.read_csv(SRC["ret_main"])
    df.columns = df.columns.str.strip()

    df["short"] = df["method"].str.strip().map(METHOD_SHORT)
    df = df.sort_values("NDCG", ascending=False).reset_index(drop=True)

    x         = np.arange(len(df))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    colors_ndcg = []
    colors_hit1 = []
    for _, row in df.iterrows():
        is_baseline = str(row["method"]).strip() == "pure_semantic_dense"
        colors_ndcg.append(BASELINE_COLOR if is_baseline else METHOD_COLORS[0])
        colors_hit1.append(BASELINE_COLOR if is_baseline else METHOD_COLORS[3])

    bars1 = ax.bar(x - bar_width / 2, df["NDCG"], bar_width,
                   label="NDCG", color=colors_ndcg, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + bar_width / 2, df["Hit@1"], bar_width,
                   label="Hit@1", color=colors_hit1, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=20, ha="right", fontsize=FONT_SIZE)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label="Dense baseline")
    ndcg_patch     = mpatches.Patch(color=METHOD_COLORS[0], label="NDCG (other methods)")
    hit1_patch     = mpatches.Patch(color=METHOD_COLORS[3], label="Hit@1 (other methods)")
    ax.legend(handles=[baseline_patch, ndcg_patch, hit1_patch],
              fontsize=FONT_SIZE - 1, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_method_comparison")


# ---------------------------------------------------------------------------
# FIGURE D — Retrieval NDCG by Entity Type (heatmap)
# ---------------------------------------------------------------------------

def make_figure_d():
    df = pd.read_csv(SRC["ret_entity"])
    df.columns = df.columns.str.strip()

    df["short"] = df["method"].str.strip().map(METHOD_SHORT)
    cols   = ["paper_NDCG", "dataset_NDCG", "model_NDCG"]
    labels = ["Paper", "Dataset", "Model"]

    data   = df[cols].values
    y_labs = df["short"].tolist()

    vmin = data.min()
    vmax = data.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_yticks(range(len(y_labs)))
    ax.set_yticklabels(y_labs, fontsize=FONT_SIZE)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            txt_color = "white" if norm(val) > 0.6 else "black"
            ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                    fontsize=FONT_SIZE - 1, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("NDCG", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    ax.set_xlabel("Entity Type", fontsize=FONT_SIZE)
    ax.set_ylabel("Retrieval Method", fontsize=FONT_SIZE)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_by_entity_type")


# ---------------------------------------------------------------------------
# FIGURE E — Retrieval NDCG by Difficulty
# ---------------------------------------------------------------------------

def make_figure_e():
    df = pd.read_csv(SRC["ret_diff"])
    df.columns = df.columns.str.strip()

    diff_cols    = ["easy_NDCG", "medium_NDCG", "hard_NDCG", "unknown_NDCG"]
    diff_labels  = ["Easy", "Medium", "Hard", "Unknown"]
    methods      = df["method"].str.strip().tolist()
    short_names  = [METHOD_SHORT.get(m, m) for m in methods]

    x         = np.arange(len(diff_labels))
    n_methods = len(methods)
    bar_width = 0.13

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (m, short, row) in enumerate(zip(methods, short_names, df.itertuples())):
        vals   = [getattr(row, c) for c in diff_cols]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        color  = METHOD_COLORS[i % len(METHOD_COLORS)]
        ax.bar(x + offset, vals, bar_width, label=short, color=color,
               alpha=0.85, edgecolor="white", linewidth=0.5)

    # Subtle hard-zone highlight
    ax.axvspan(1.5, 2.5, alpha=0.06, color="red")
    ax.annotate("Hard\nquestions", xy=(2, 0.03), ha="center",
                fontsize=FONT_SIZE - 2, color="darkred")

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.set_ylabel("NDCG")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=FONT_SIZE - 1, ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.18), loc="upper center")

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_by_difficulty")


# ---------------------------------------------------------------------------
# FIGURE F — Question-Type NDCG Heatmap
# ---------------------------------------------------------------------------

def make_figure_f():
    df = pd.read_csv(SRC["ret_qtype"])
    df.columns = df.columns.str.strip()

    method_order = [
        "pure_semantic_dense", "hybrid_type_filtering", "hybrid_type_onehop_filtering",
        "hybrid_predicate_aware_filtering", "optional_rrf_fusion", "optional_rrf_symbolic",
    ]
    short_methods = [METHOD_SHORT.get(m, m) for m in method_order]

    # Filter: max count across methods >= 3
    max_counts = df.groupby("question_type")["count"].max()
    valid_qtypes = max_counts[max_counts >= 3].index.tolist()

    if not valid_qtypes:
        warnings_list.append("FIGURE F: No question types with count >= 3; skipping heatmap")
        return

    # Sort by dense NDCG descending
    dense_ndcg = (df[df["method"].str.strip() == "pure_semantic_dense"]
                  .set_index("question_type")["NDCG"])
    valid_qtypes_sorted = sorted(
        valid_qtypes,
        key=lambda qt: -dense_ndcg.get(qt, 0),
    )

    data = np.full((len(valid_qtypes_sorted), len(method_order)), np.nan)
    for row_i, qt in enumerate(valid_qtypes_sorted):
        for col_j, m in enumerate(method_order):
            sub = df[(df["question_type"].str.strip() == qt) &
                     (df["method"].str.strip() == m)]
            if not sub.empty:
                data[row_i, col_j] = float(sub.iloc[0]["NDCG"])

    yticklabels = [
        QTYPE_READABLE_PLAIN.get(qt, qt.replace("_", " ").title())
        for qt in valid_qtypes_sorted
    ]

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(8, max(5, len(valid_qtypes_sorted) * 0.48)))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(short_methods)))
    ax.set_xticklabels(short_methods, rotation=30, ha="right", fontsize=FONT_SIZE)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Retrieval Method", fontsize=FONT_SIZE)
    ax.set_ylabel("Question Type", fontsize=FONT_SIZE)

    # Cell annotations (2 decimal places)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if not np.isnan(val):
                txt_color = "black" if 0.35 < norm(val) < 0.75 else "white"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=FONT_SIZE - 2, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("NDCG", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_question_type_heatmap")


# ---------------------------------------------------------------------------
# FIGURE G — Hit@1 vs Hit@10 Recoverability Gap
# ---------------------------------------------------------------------------

def make_figure_g():
    df = pd.read_csv(SRC["ret_gap"])
    df.columns = df.columns.str.strip()

    df["short"] = df["method"].str.strip().map(METHOD_SHORT)
    gap_col = "gap_Hit10_minus_Hit1"

    x         = np.arange(len(df))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (_, row) in enumerate(df.iterrows()):
        is_baseline = str(row["method"]).strip() == "pure_semantic_dense"
        c1 = BASELINE_COLOR if is_baseline else "#4472C4"
        c2 = BASELINE_COLOR if is_baseline else "#A8BDD8"

        ax.bar(x[i] - bar_width / 2, row["Hit@1"],  bar_width, color=c1,
               alpha=0.9, edgecolor="white", label="Hit@1"  if i == 1 else "_nolegend_")
        ax.bar(x[i] + bar_width / 2, row["Hit@10"], bar_width, color=c2,
               alpha=0.9, edgecolor="white", label="Hit@10" if i == 1 else "_nolegend_")

        gap = float(row[gap_col])
        ax.annotate(
            f"+{gap:.3f}",
            xy=(x[i], max(row["Hit@1"], row["Hit@10"]) + 0.015),
            ha="center", va="bottom", fontsize=FONT_SIZE - 2, color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=20, ha="right", fontsize=FONT_SIZE)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    hit1_patch  = mpatches.Patch(color="#4472C4",    label="Hit@1")
    hit10_patch = mpatches.Patch(color="#A8BDD8",    label="Hit@10")
    base_patch  = mpatches.Patch(color=BASELINE_COLOR, label="Dense baseline")
    ax.legend(handles=[hit1_patch, hit10_patch, base_patch],
              fontsize=FONT_SIZE - 1, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig_retrieval_hit_gap")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_ENTRIES = [
    {
        "type": "table", "file": "tab_evaluation_dataset_summary.tex",
        "source": "data/questions/ml_questions_dataset.json + data/results/retrieval/summary_by_difficulty.json",
        "section": "Section 4.1 (Evaluation Setup)",
        "label": "tab:evaluation_dataset_summary",
        "caption": "Evaluation dataset summary.",
        "note": "Characterises the 280-question benchmark. Counts for paper, dataset, and model "
                "questions are derived from the question_type field. Difficulty distribution "
                "comes from the retrieval evaluation (265 answerable questions).",
    },
    {
        "type": "table", "file": "tab_best_representation_by_entity.tex",
        "source": "data/results/thesis_tables/best_per_entity.csv",
        "section": "Section 4.2 (Pre-Retrieval Evaluation)",
        "label": "tab:best_representation_by_entity",
        "caption": "Best entity-centric representation strategy per entity type, evaluated at top-10 candidate entities.",
        "note": "Shows the single best strategy for each entity type. Paper benefits from enriched "
                "metadata; models are best served by predicate-filtered text; datasets remain "
                "challenging even with the best strategy.",
    },
    {
        "type": "table", "file": "tab_pre_retrieval_full_comparison.tex",
        "source": "data/results/thesis_tables/full_comparison.csv",
        "section": "Section 4.2",
        "label": "tab:pre_retrieval_full_comparison",
        "caption": "Pre-retrieval representation comparison across all 14 strategies and three entity types.",
        "note": "All 14 strategies shown, grouped by entity type with NDCG descending. The spread "
                "within each group indicates how sensitive each entity type is to representation choice.",
    },
    {
        "type": "table", "file": "tab_pre_retrieval_difficulty.tex",
        "source": "data/results/thesis_tables/difficulty_breakdown.csv",
        "section": "Section 4.2",
        "label": "tab:pre_retrieval_difficulty",
        "caption": "Difficulty breakdown for the best-performing representation per entity type.",
        "note": "Filters to only the winning strategy per entity type. Reveals that hard questions "
                "depress performance significantly regardless of entity type.",
    },
    {
        "type": "table", "file": "tab_retrieval_method_comparison.tex",
        "source": "data/results/retrieval/thesis_tables/retrieval_main_comparison.csv",
        "section": "Section 4.3 (Retrieval Evaluation)",
        "label": "tab:retrieval_method_comparison",
        "caption": "Retrieval method comparison across six methods evaluated on 265 questions over top-10 candidate entities.",
        "note": "RRF+Symbolic achieves the best overall NDCG (+0.0097 over dense). Hybrid methods "
                "produce smaller but consistent gains. Hit@1 is highest for Hybrid-Predicate-Aware.",
    },
    {
        "type": "table", "file": "tab_retrieval_by_entity_type.tex",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv",
        "section": "Section 4.3",
        "label": "tab:retrieval_by_entity_type",
        "caption": "Retrieval NDCG broken down by entity type.",
        "note": "Dataset entities remain the hardest category; RRF fusion improves dataset NDCG "
                "from 0.3822 to 0.4645. Paper and model NDCG are relatively stable across methods.",
    },
    {
        "type": "table", "file": "tab_retrieval_by_difficulty.tex",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv",
        "section": "Section 4.3",
        "label": "tab:retrieval_by_difficulty",
        "caption": "Retrieval NDCG broken down by question difficulty.",
        "note": "Hard questions (NDCG ~0.47-0.50) are the clear bottleneck. Easy questions are near "
                "ceiling. RRF methods improve medium and unknown categories.",
    },
    {
        "type": "table", "file": "tab_retrieval_hit_gap.tex",
        "source": "data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv",
        "section": "Section 4.3",
        "label": "tab:retrieval_hit_gap",
        "caption": "Recoverability gap between Hit@1 and Hit@10 across retrieval methods.",
        "note": "RRF-Fusion shows the largest gap (0.1698), meaning it retrieves more correct "
                "candidate entities in the top-10 but ranks them lower than the dense baseline does.",
    },
    {
        "type": "table", "file": "tab_retrieval_question_type_summary.tex",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv",
        "section": "Section 4.3",
        "label": "tab:retrieval_question_type_summary",
        "caption": "Question-type retrieval summary: selected question types representing best-performing, worst-performing, and RRF-improved categories.",
        "note": "Compact view of 9 representative question types. Full 22-type heatmap is shown in "
                "Figure~\\ref{fig:retrieval_question_type_heatmap}.",
    },
    {
        "type": "figure", "file": "fig_pre_retrieval_ndcg_by_representation.pdf / .png",
        "source": "data/results/thesis_tables/full_comparison.csv",
        "section": "Section 4.2",
        "label": "fig:pre_retrieval_ndcg_by_representation",
        "caption": "NDCG of all pre-retrieval representation strategies, grouped by entity type. The best strategy per group is highlighted.",
        "note": "3-panel horizontal bar chart. Consolidates the six per-entity figures previously "
                "in data/results/thesis_figures/ into one compact thesis-ready figure. Regenerated "
                "for consistent styling.",
        "reuse": "Not reused. Existing figures (ndcg_paper/dataset/model) are separate panels; "
                 "this new figure combines them with consistent style.",
    },
    {
        "type": "figure", "file": "fig_pre_retrieval_difficulty_ndcg.pdf / .png",
        "source": "data/results/thesis_tables/difficulty_breakdown.csv",
        "section": "Section 4.2",
        "label": "fig:pre_retrieval_difficulty_ndcg",
        "caption": "NDCG by question difficulty for the best representation per entity type.",
        "note": "Shows how hard questions depress performance across all entity types. Dataset "
                "questions are hard at all difficulty levels.",
        "reuse": "Not reused. Equivalent to best_repr_difficulty_breakdown_ndcg.png but regenerated "
                 "with consistent font/color styling.",
    },
    {
        "type": "figure", "file": "fig_retrieval_method_comparison.pdf / .png",
        "source": "data/results/retrieval/thesis_tables/retrieval_main_comparison.csv",
        "section": "Section 4.3",
        "label": "fig:retrieval_method_comparison",
        "caption": "NDCG and Hit@1 for all six retrieval methods. The dense semantic baseline is shown in grey.",
        "note": "Shows both the ranking quality (NDCG) and top-1 precision (Hit@1) side by side. "
                "New artifact — no equivalent in thesis_figures/.",
        "reuse": "No existing equivalent.",
    },
    {
        "type": "figure", "file": "fig_retrieval_by_entity_type.pdf / .png",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv",
        "section": "Section 4.3",
        "label": "fig:retrieval_by_entity_type",
        "caption": "Heatmap of retrieval NDCG by entity type and method.",
        "note": "Reveals that dataset retrieval is weakest and benefits most from RRF fusion. "
                "Paper and model NDCG are stable across methods.",
        "reuse": "No existing equivalent.",
    },
    {
        "type": "figure", "file": "fig_retrieval_by_difficulty.pdf / .png",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv",
        "section": "Section 4.3",
        "label": "fig:retrieval_by_difficulty",
        "caption": "Retrieval NDCG by question difficulty across all six methods.",
        "note": "Emphasises that hard questions are the bottleneck. RRF methods marginally improve "
                "medium and unknown difficulty while not recovering hard questions significantly.",
        "reuse": "No existing equivalent.",
    },
    {
        "type": "figure", "file": "fig_retrieval_question_type_heatmap.pdf / .png",
        "source": "data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv",
        "section": "Section 4.3",
        "label": "fig:retrieval_question_type_heatmap",
        "caption": "Heatmap of NDCG by question type and retrieval method. Question types with fewer than 3 questions are excluded.",
        "note": "Filtered to N>=3 question types (excludes 3 low-count types: model_name_resolution, "
                "dataset_to_publication_year, semantic_repository_to_model). Sorted by dense NDCG "
                "descending. Reveals which question types benefit from RRF fusion.",
        "reuse": "No existing equivalent.",
    },
    {
        "type": "figure", "file": "fig_retrieval_hit_gap.pdf / .png",
        "source": "data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv",
        "section": "Section 4.3",
        "label": "fig:retrieval_hit_gap",
        "caption": "Hit@1 and Hit@10 for each retrieval method, with the recoverability gap annotated above each method pair.",
        "note": "Supports the argument that RRF fusion sacrifices top-1 precision but recovers "
                "more correct candidate entities by rank 10.",
        "reuse": "No existing equivalent.",
    },
]


def make_manifest():
    lines = []
    lines.append("# Chapter 4 Artifact Manifest")
    lines.append("")
    lines.append("All artifacts cover **pre-retrieval (Section 4.2)** and "
                 "**retrieval (Section 4.3)** stages only.")
    lines.append("No post-retrieval artifacts are included.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Tables")
    lines.append("")

    for entry in MANIFEST_ENTRIES:
        if entry["type"] != "table":
            continue
        lines.append(f"### `{entry['file']}`")
        lines.append(f"- **Source:** `{entry['source']}`")
        lines.append(f"- **Section:** {entry['section']}")
        lines.append(f"- **LaTeX label:** `\\label{{{entry['label']}}}`")
        lines.append(f"- **Suggested caption:** {entry['caption']}")
        lines.append(f"- **Interpretation:** {entry['note']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Figures")
    lines.append("")

    for entry in MANIFEST_ENTRIES:
        if entry["type"] != "figure":
            continue
        lines.append(f"### `{entry['file']}`")
        lines.append(f"- **Source:** `{entry['source']}`")
        lines.append(f"- **Section:** {entry['section']}")
        lines.append(f"- **LaTeX label:** `\\label{{{entry['label']}}}`")
        lines.append(f"- **Suggested caption:** {entry['caption']}")
        lines.append(f"- **Interpretation:** {entry['note']}")
        lines.append(f"- **Reuse decision:** {entry.get('reuse', 'N/A')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Method Name Mapping")
    lines.append("")
    lines.append("| Short (figure axes) | Display (table rows) | Method ID |")
    lines.append("|---|---|---|")
    for k, short in METHOD_SHORT.items():
        lines.append(f"| {short} | {METHOD_DISPLAY[k]} | `{k}` |")

    lines.append("")
    lines.append("## Figure Filtering Rules")
    lines.append("")
    lines.append("- **Figure F (heatmap):** Excludes question types with max count < 3 "
                 "(`model_name_resolution` N=1, `dataset_to_publication_year` N=2, "
                 "`semantic_repository_to_model` N=2).")
    lines.append("")

    manifest_path = REPO_ROOT / "docs" / "chapter4_artifacts" / "chapter4_artifact_manifest.md"
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [MANIFEST] chapter4_artifact_manifest.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Chapter 4 Artifact Generator")
    print("=" * 60)

    print("\n[1/3] Validating inputs ...")
    validate_inputs()
    print("  All source files found.")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Generating tables ...")
    make_table_a()
    make_table_b()
    make_table_c()
    make_table_d()
    make_table_e()
    make_table_f()
    make_table_g()
    make_table_h()
    make_table_i()

    print("\n[3/3] Generating figures ...")
    make_figure_a()
    make_figure_b()
    make_figure_c()
    make_figure_d()
    make_figure_e()
    make_figure_f()
    make_figure_g()

    make_manifest()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTables generated ({len(generated_tables)}):")
    for t in generated_tables:
        print(f"  docs/chapter4_artifacts/tables/{t}")
    print(f"\nFigures generated ({len(generated_figures)}):")
    for f in generated_figures:
        print(f"  docs/chapter4_artifacts/figures/{f}")
    print(f"\nManifest: docs/chapter4_artifacts/chapter4_artifact_manifest.md")

    if warnings_list:
        print(f"\nWarnings ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"  ! {w}")
    else:
        print("\nNo warnings.")


if __name__ == "__main__":
    main()
