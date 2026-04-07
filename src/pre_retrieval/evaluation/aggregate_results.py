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
METRIC_NAMES = ("Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG")


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
    representation = payload.get("representation_type", result_file.stem.replace("_results", ""))
    return {
        "representation": representation,
        **{metric_name: float(summary.get(metric_name, 0.0)) for metric_name in METRIC_NAMES},
    }


def aggregate_result_files(output_dir: Path, representation_order: Sequence[str] | None = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    order_map = _representation_order_map(representation_order or [])
    rows = [_extract_summary_row(result_file) for result_file in output_dir.glob("*_results.json")]
    rows.sort(key=lambda row: (order_map.get(row["representation"], len(order_map)), row["representation"]))

    summary_payload = {"rows": rows}
    save_json(summary_payload, output_dir / SUMMARY_FILE_NAME)
    (output_dir / SUMMARY_MARKDOWN_FILE_NAME).write_text(_build_summary_markdown(rows), encoding="utf-8")
    _write_summary_csv(rows, output_dir / SUMMARY_CSV_FILE_NAME)
    return summary_payload


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
