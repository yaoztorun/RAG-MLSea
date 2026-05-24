#!/usr/bin/env python3
"""
generate_chapter4_tables.py

Generates Chapter 4 LaTeX tables from current (500-question) evaluation results.

Outputs files with _current suffix to docs/chapter4_artifacts/tables/.
Never touches existing non-_current files.

Usage:
    python scripts/generate_chapter4_tables.py
    python scripts/generate_chapter4_tables.py --force   # regenerate even if output exists

Sources (all from 500-question evaluation):
    data/questions/ml_questions_dataset.json
    data/results/pre_retrieval/summary.csv
    data/results/retrieval/summary_by_difficulty.json   (for difficulty counts)
    data/results/retrieval/thesis_tables/retrieval_main_comparison.csv
    data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv
    data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_QUESTIONS   = REPO_ROOT / "data" / "questions" / "ml_questions_dataset.json"
SRC_PRE_SUMMARY = REPO_ROOT / "data" / "results" / "pre_retrieval" / "summary.csv"
SRC_RET_DIFF_JSON = REPO_ROOT / "data" / "results" / "retrieval" / "summary_by_difficulty.json"
SRC_RET_MAIN    = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_main_comparison.csv"
SRC_RET_ENTITY  = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_entity_type_ndcg.csv"
SRC_RET_DIFF    = REPO_ROOT / "data" / "results" / "retrieval" / "thesis_tables" / "retrieval_by_difficulty_ndcg.csv"

OUT_DIR = REPO_ROOT / "docs" / "chapter4_artifacts" / "tables"

# ---------------------------------------------------------------------------
# Method display names
# ---------------------------------------------------------------------------

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

ENTITY_DISPLAY = {"paper": "Paper", "dataset": "Dataset", "model": "Model"}

REPR_DISPLAY = {
    # Paper
    "title_only":           "Title Only",
    "abstract_only":        "Abstract Only",
    "title_abstract":       "Title + Abstract",
    "predicate_filtered":   "Predicate-Filtered",
    "enriched_metadata":    "Enriched Metadata",
    "one_hop":              "One-Hop",
    # Dataset
    "dataset_title_only":          "Title Only",
    "dataset_metadata":            "Metadata",
    "dataset_predicate_filtered":  "Predicate-Filtered",
    "dataset_enriched_metadata":   "Enriched Metadata",
    # Model
    "model_title_only":          "Title Only",
    "model_metadata":            "Metadata",
    "model_predicate_filtered":  "Predicate-Filtered",
    "model_enriched_metadata":   "Enriched Metadata",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

generated: list[tuple[str, Path]] = []
skipped: list[tuple[str, Path]] = []


def fmt(val: float, digits: int = 4) -> str:
    if pd.isna(val):
        return "--"
    return f"{float(val):.{digits}f}"


def fmt_delta(val: float, digits: int = 4) -> str:
    if pd.isna(val):
        return "--"
    v = float(val)
    if abs(v) < 1e-9:
        return f"0.{('0' * digits)}"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{digits}f}"


def esc(s: str) -> str:
    return (str(s)
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#"))


def save_table(tex: str, filename: str, force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    if path.exists() and not force:
        skipped.append((filename, path))
        print(f"  [SKIP]  {filename}  (already exists; use --force to overwrite)")
        return
    path.write_text(tex, encoding="utf-8")
    generated.append((filename, path))
    print(f"  [TABLE] {filename}")


def validate():
    required = [SRC_QUESTIONS, SRC_PRE_SUMMARY, SRC_RET_DIFF_JSON,
                SRC_RET_MAIN, SRC_RET_ENTITY, SRC_RET_DIFF]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("ERROR: Missing source files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# TABLE 1 — Evaluation dataset statistics
# ---------------------------------------------------------------------------

def make_table_evaluation_dataset_summary(force: bool) -> None:
    questions = json.load(open(SRC_QUESTIONS, encoding="utf-8"))
    answerable   = [q for q in questions if q.get("is_answerable", True)]
    unanswerable = [q for q in questions if not q.get("is_answerable", True)]

    n_paper   = sum(1 for q in answerable if q.get("target_entity_type") == "paper")
    n_dataset = sum(1 for q in answerable if q.get("target_entity_type") == "dataset")
    n_model   = sum(1 for q in answerable if q.get("target_entity_type") == "model")

    diff_data = json.load(open(SRC_RET_DIFF_JSON, encoding="utf-8"))
    dense = diff_data.get("pure_semantic_dense", {})
    n_easy   = dense.get("easy",   {}).get("count", "?")
    n_medium = dense.get("medium", {}).get("count", "?")
    n_hard   = dense.get("hard",   {}).get("count", "?")

    # Verification
    assert len(questions) == 520, f"Expected 520 total, got {len(questions)}"
    assert len(answerable) == 500, f"Expected 500 answerable, got {len(answerable)}"
    assert n_paper == 250, f"Expected 250 paper, got {n_paper}"
    assert n_dataset == 125, f"Expected 125 dataset, got {n_dataset}"
    assert n_model == 125, f"Expected 125 model, got {n_model}"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{Evaluation dataset statistics. Difficulty counts apply to the"
        r" answerable subset used in quantitative evaluation.}",
        r"  \label{tab:evaluation_dataset_summary}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Property} & \textbf{Count} \\",
        r"    \midrule",
        rf"    Total questions & {len(questions)} \\",
        rf"    Answerable (used in evaluation) & {len(answerable)} \\",
        rf"    Unanswerable (excluded from metric averages) & {len(unanswerable)} \\",
        r"    \addlinespace",
        rf"    Paper questions & {n_paper} \\",
        rf"    Dataset questions & {n_dataset} \\",
        rf"    Model questions & {n_model} \\",
        r"    \addlinespace",
        r"    \multicolumn{2}{l}{\textit{Difficulty (answerable questions)}} \\",
        rf"    \quad Easy & {n_easy} \\",
        rf"    \quad Medium & {n_medium} \\",
        rf"    \quad Hard & {n_hard} \\",
        r"    \addlinespace",
        r"    \multicolumn{2}{l}{\textit{Evaluation metrics}} \\",
        r"    \quad Hit@1, Hit@5, Hit@10, MRR, NDCG & (primary: NDCG) \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_evaluation_dataset_summary_current.tex", force)


# ---------------------------------------------------------------------------
# TABLE 2 — Best representation per entity type (compact, 3 rows)
# ---------------------------------------------------------------------------

def make_table_best_representation(force: bool) -> None:
    df = pd.read_csv(SRC_PRE_SUMMARY)
    df.columns = df.columns.str.strip()

    best_rows = []
    for et in ["paper", "dataset", "model"]:
        group = df[df["entity_type"].str.strip() == et]
        best = group.loc[group["NDCG"].idxmax()]
        best_rows.append(best)

    # Spot-check expected values
    best_dict = {r["entity_type"].strip(): r for r in best_rows}
    assert abs(float(best_dict["paper"]["NDCG"]) - 0.4544) < 0.001, \
        f"Paper best NDCG mismatch: {best_dict['paper']['NDCG']}"
    assert abs(float(best_dict["dataset"]["NDCG"]) - 0.5281) < 0.001, \
        f"Dataset best NDCG mismatch: {best_dict['dataset']['NDCG']}"
    assert abs(float(best_dict["model"]["NDCG"]) - 0.5046) < 0.001, \
        f"Model best NDCG mismatch: {best_dict['model']['NDCG']}"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        r"  \caption{Best representation strategy per entity type evaluated over"
        r" top-10 candidate entities on 500 answerable questions. NDCG is the"
        r" primary ranking metric.}",
        r"  \label{tab:best_representation_by_entity}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{Entity Type} & \textbf{Representation}"
        r" & \textbf{Hit@1} & \textbf{Hit@10} & \textbf{NDCG} \\",
        r"    \midrule",
    ]
    for row in best_rows:
        et = row["entity_type"].strip()
        rep = row["representation"].strip()
        rep_label = REPR_DISPLAY.get(rep, rep.replace("_", " ").title())
        et_label = ENTITY_DISPLAY.get(et, et.title())
        lines.append(
            f"    {et_label} & {rep_label}"
            f" & {fmt(row['Hit@1'])} & {fmt(row['Hit@10'])} & \\textbf{{{fmt(row['NDCG'])}}} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_best_representation_by_entity_current.tex", force)


# ---------------------------------------------------------------------------
# TABLE 3 — Full pre-retrieval comparison (14 rows, appendix-ready)
# ---------------------------------------------------------------------------

def make_table_pre_retrieval_full(force: bool) -> None:
    df = pd.read_csv(SRC_PRE_SUMMARY)
    df.columns = df.columns.str.strip()

    ENTITY_ORDER = ["paper", "dataset", "model"]
    METRICS = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]

    lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{multirow}",
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{Pre-retrieval representation comparison across all 14 strategies"
        r" and three entity types, evaluated over 500 answerable questions at top-10"
        r" candidates. Bold denotes the highest NDCG within each entity-type group.}",
        r"  \label{tab:appendix_pre_retrieval_full_comparison}",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    \textbf{Entity Type} & \textbf{Representation}"
        r" & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10}"
        r" & \textbf{MRR} & \textbf{NDCG} \\",
        r"    \midrule",
    ]

    first = True
    for et in ENTITY_ORDER:
        group = df[df["entity_type"].str.strip() == et].copy()
        group = group.sort_values("NDCG", ascending=False).reset_index(drop=True)
        best_ndcg = float(group["NDCG"].max())

        if not first:
            lines.append(r"    \midrule")
        first = False

        n = len(group)
        for i, (_, row) in enumerate(group.iterrows()):
            rep = row["representation"].strip()
            rep_label = REPR_DISPLAY.get(rep, rep.replace("_", " ").title())
            ndcg_val = float(row["NDCG"])
            ndcg_str = fmt(ndcg_val)
            if abs(ndcg_val - best_ndcg) < 1e-9:
                ndcg_str = r"\textbf{" + ndcg_str + r"}"

            metric_cells = [fmt(row[m]) for m in METRICS[:-1]] + [ndcg_str]
            vals = " & ".join(metric_cells)

            if i == 0:
                et_cell = r"\multirow{" + str(n) + r"}{*}{" + ENTITY_DISPLAY.get(et, et.title()) + r"}"
            else:
                et_cell = ""
            lines.append(f"    {et_cell} & {rep_label} & {vals} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_pre_retrieval_full_comparison_current.tex", force)


# ---------------------------------------------------------------------------
# TABLE 4 — Retrieval method comparison
# ---------------------------------------------------------------------------

def make_table_retrieval_method_comparison(force: bool) -> None:
    df = pd.read_csv(SRC_RET_MAIN)
    df.columns = df.columns.str.strip()

    # Spot-check
    dense_row = df[df["method"].str.strip() == "pure_semantic_dense"].iloc[0]
    assert abs(float(dense_row["NDCG"]) - 0.4854) < 0.001, \
        f"Dense NDCG mismatch: {dense_row['NDCG']}"
    rrf_row = df[df["method"].str.strip() == "optional_rrf_fusion"].iloc[0]
    assert abs(float(rrf_row["NDCG"]) - 0.4962) < 0.001, \
        f"RRF NDCG mismatch: {rrf_row['NDCG']}"

    best_ndcg = float(df["NDCG"].max())
    METRICS = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        r"  \caption{Retrieval method comparison across six methods evaluated on"
        r" 500 answerable questions at top-10 candidates."
        r" $\Delta$NDCG is relative to the dense semantic baseline."
        r" \textsuperscript{\dag}~Dense semantic baseline.}",
        r"  \label{tab:retrieval_method_comparison}",
        r"  \begin{tabular}{llcccccr}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{Group} & \textbf{Hit@1} & \textbf{Hit@5}"
        r" & \textbf{Hit@10} & \textbf{MRR} & \textbf{NDCG} & \textbf{$\Delta$NDCG} \\",
        r"    \midrule",
    ]

    for _, row in df.iterrows():
        method = str(row["method"]).strip()
        display = esc(METHOD_DISPLAY.get(method, method))
        group   = esc(METHOD_GROUP.get(method, ""))
        delta   = fmt_delta(row.get("delta_NDCG_vs_dense", 0))

        ndcg_val = float(row["NDCG"])
        ndcg_str = fmt(ndcg_val)
        if abs(ndcg_val - best_ndcg) < 1e-9:
            ndcg_str = r"\textbf{" + ndcg_str + r"}"

        metric_cells = [fmt(row[m]) for m in METRICS[:-1]] + [ndcg_str]
        vals = " & ".join(metric_cells)

        dag = r"\textsuperscript{\dag}" if method == "pure_semantic_dense" else ""
        lines.append(f"    {display}{dag} & {group} & {vals} & {delta} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_retrieval_method_comparison_current.tex", force)


# ---------------------------------------------------------------------------
# TABLE 5 — Retrieval NDCG by entity type
# ---------------------------------------------------------------------------

def make_table_retrieval_by_entity_type(force: bool) -> None:
    df = pd.read_csv(SRC_RET_ENTITY)
    df.columns = df.columns.str.strip()

    cols = ["paper_NDCG", "dataset_NDCG", "model_NDCG"]
    best = {c: float(df[c].max()) for c in cols}

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        r"  \caption{Retrieval NDCG segmented by entity type."
        r" Bold denotes the highest NDCG per column.}",
        r"  \label{tab:retrieval_by_entity_type}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{Paper} & \textbf{Dataset} & \textbf{Model} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        method = str(row["method"]).strip()
        display = esc(METHOD_DISPLAY.get(method, method))
        cells = []
        for c in cols:
            s = fmt(row[c])
            if abs(float(row[c]) - best[c]) < 1e-9:
                s = r"\textbf{" + s + r"}"
            cells.append(s)
        lines.append(f"    {display} & {' & '.join(cells)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_retrieval_by_entity_type_current.tex", force)


# ---------------------------------------------------------------------------
# TABLE 6 — Retrieval NDCG by difficulty (exclude unknown column)
# ---------------------------------------------------------------------------

def make_table_retrieval_by_difficulty(force: bool) -> None:
    df = pd.read_csv(SRC_RET_DIFF)
    df.columns = df.columns.str.strip()

    cols = ["easy_NDCG", "medium_NDCG", "hard_NDCG"]
    headers = ["Easy", "Medium", "Hard"]
    best = {c: float(df[c].max()) for c in cols}

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        r"  \caption{Retrieval NDCG segmented by question difficulty."
        r" Bold denotes the highest NDCG per column."
        r" Unknown-difficulty questions are excluded.}",
        r"  \label{tab:retrieval_by_difficulty}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        method = str(row["method"]).strip()
        display = esc(METHOD_DISPLAY.get(method, method))
        cells = []
        for c in cols:
            s = fmt(row[c])
            if abs(float(row[c]) - best[c]) < 1e-9:
                s = r"\textbf{" + s + r"}"
            cells.append(s)
        lines.append(f"    {display} & {' & '.join(cells)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    save_table("\n".join(lines), "tab_retrieval_by_difficulty_current.tex", force)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 LaTeX tables from current (500-question) results."
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing _current files.")
    args = parser.parse_args()

    print("=" * 60)
    print("Chapter 4 Table Generator  (500-question dataset)")
    print("=" * 60)

    print("\nValidating source files ...")
    validate()
    print("  All source files present.")

    print("\nGenerating tables ...")
    make_table_evaluation_dataset_summary(args.force)
    make_table_best_representation(args.force)
    make_table_pre_retrieval_full(args.force)
    make_table_retrieval_method_comparison(args.force)
    make_table_retrieval_by_entity_type(args.force)
    make_table_retrieval_by_difficulty(args.force)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if generated:
        print(f"\nGenerated ({len(generated)}):")
        for name, path in generated:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}) — already current:")
        for name, path in skipped:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}")
    if not generated and not skipped:
        print("  Nothing to do.")


if __name__ == "__main__":
    main()
