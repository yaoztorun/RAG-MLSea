from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.utils import load_json, save_json


SUMMARY_FILE_NAME = "summary.json"
SUMMARY_MARKDOWN_FILE_NAME = "summary.md"
SUMMARY_CSV_FILE_NAME = "summary.csv"
SUMMARY_BY_DIFFICULTY_FILE_NAME = "summary_by_difficulty.json"
SUMMARY_BY_CATEGORY_FILE_NAME = "summary_by_category.json"
RESULTS_FILE_NAME = "results.json"
METRIC_NAMES = ("Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG")
DIFFICULTY_ORDER = ("easy", "medium", "hard")
CATEGORY_ORDER = ("paper", "dataset", "implementation", "multihop", "semantic", "unanswerable")


def _representation_order_map(representation_order: Sequence[str]) -> Dict[str, int]:
    return {representation: index for index, representation in enumerate(representation_order)}


def _build_summary_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Representation | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG |",
        "|----------------|------:|------:|-------:|------:|------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation']} | {row['Hit@1']:.4f} | {row['Hit@5']:.4f} | {row['Hit@10']:.4f} | {row['MRR']:.4f} | {row['NDCG']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["representation", *METRIC_NAMES])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "representation": row["representation"],
                    **{metric_name: f"{row[metric_name]:.4f}" for metric_name in METRIC_NAMES},
                }
            )


def _extract_summary_row(result_file: Path) -> Dict[str, Any]:
    payload = load_json(result_file)
    summary = payload.get("metrics", {})
    representation = payload.get("representation_type", result_file.parent.name or result_file.stem.replace("_results", ""))
    return {
        "representation": representation,
        **{metric_name: float(summary.get(metric_name, 0.0)) for metric_name in METRIC_NAMES},
    }


def _extract_segment_rows(result_file: Path, section_name: str, segment_name: str) -> List[Dict[str, Any]]:
    payload = load_json(result_file)
    representation = payload.get("representation_type", result_file.parent.name or result_file.stem.replace("_results", ""))
    segment_payload = payload.get(section_name, {})
    rows: List[Dict[str, Any]] = []
    for segment_value, segment_summary in segment_payload.items():
        row = {
            segment_name: segment_value,
            "representation": representation,
            **{metric_name: float(segment_summary.get(metric_name, 0.0)) for metric_name in METRIC_NAMES},
        }
        for count_name in (
            "total_questions",
            "answerable_questions",
            "paper_target_questions",
            "evaluated_questions",
            "skipped_questions",
            "skipped_non_paper_targets",
            "skipped_unanswerable",
        ):
            if count_name in segment_summary:
                row[count_name] = int(segment_summary.get(count_name, 0))
        if "unanswerable_rejection_rate" in segment_summary:
            row["unanswerable_rejection_rate"] = float(segment_summary.get("unanswerable_rejection_rate", 0.0))
        if "false_accept_rate" in segment_summary:
            row["false_accept_rate"] = float(segment_summary.get("false_accept_rate", 0.0))
        rows.append(row)
    return rows


def _aggregate_segment_rows(
    result_files: Sequence[Path],
    *,
    section_name: str,
    segment_name: str,
    representation_order: Sequence[str],
) -> Dict[str, Any]:
    order_map = _representation_order_map(representation_order)
    segments: Dict[str, List[Dict[str, Any]]] = {}
    for result_file in result_files:
        for row in _extract_segment_rows(result_file, section_name, segment_name):
            segments.setdefault(str(row[segment_name]), []).append(row)

    for rows in segments.values():
        rows.sort(key=lambda row: (order_map.get(row["representation"], len(order_map)), row["representation"]))

    preferred_segment_order = DIFFICULTY_ORDER if segment_name == "difficulty" else CATEGORY_ORDER
    preferred_order_map = {segment: index for index, segment in enumerate(preferred_segment_order)}
    ordered_segments = {
        segment: {"rows": segments[segment]}
        for segment in sorted(segments, key=lambda value: (preferred_order_map.get(value, len(preferred_order_map)), value))
    }
    return {"segments": ordered_segments}


def _discover_result_files(output_dir: Path) -> List[Path]:
    nested_results = sorted(output_dir.glob(f"*/{RESULTS_FILE_NAME}"))
    if nested_results:
        return nested_results
    return sorted(output_dir.glob("*_results.json"))


def aggregate_result_files(output_dir: Path, representation_order: Sequence[str] | None = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_representations = representation_order or []
    order_map = _representation_order_map(ordered_representations)
    result_files = _discover_result_files(output_dir)
    rows = [_extract_summary_row(result_file) for result_file in result_files]
    rows.sort(key=lambda row: (order_map.get(row["representation"], len(order_map)), row["representation"]))

    summary_payload = {"rows": rows}
    summary_by_difficulty_payload = _aggregate_segment_rows(
        result_files,
        section_name="metrics_by_difficulty",
        segment_name="difficulty",
        representation_order=ordered_representations,
    )
    summary_by_category_payload = _aggregate_segment_rows(
        result_files,
        section_name="metrics_by_category",
        segment_name="category",
        representation_order=ordered_representations,
    )
    save_json(summary_payload, output_dir / SUMMARY_FILE_NAME)
    save_json(summary_by_difficulty_payload, output_dir / SUMMARY_BY_DIFFICULTY_FILE_NAME)
    save_json(summary_by_category_payload, output_dir / SUMMARY_BY_CATEGORY_FILE_NAME)
    (output_dir / SUMMARY_MARKDOWN_FILE_NAME).write_text(_build_summary_markdown(rows), encoding="utf-8")
    _write_summary_csv(rows, output_dir / SUMMARY_CSV_FILE_NAME)
    return {
        **summary_payload,
        "summary_by_difficulty": summary_by_difficulty_payload,
        "summary_by_category": summary_by_category_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate retrieval result files into summary outputs.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    output_dir = resolve_repo_path(args.output_dir or config["evaluation"]["output_dir"])
    aggregate_result_files(output_dir=output_dir, representation_order=config["evaluation"]["representation_order"])
    print(f"Wrote aggregate summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
