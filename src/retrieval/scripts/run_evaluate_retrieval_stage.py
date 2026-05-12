"""Evaluate one or all retrieval-stage methods from saved results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.config import ALL_METHODS, METHOD_ALIASES, RETRIEVAL_OUTPUT_DIR

_ALL_CHOICES = sorted(set(ALL_METHODS) | set(METHOD_ALIASES.keys()))
from src.retrieval.evaluate_retrieval_stage import evaluate_results
from src.retrieval.aggregate_retrieval_stage import write_json


def _evaluate_method(method_name: str, output_dir: Path, verbose: bool = True) -> bool:
    results_path = output_dir / method_name / "results.json"
    metrics_path = output_dir / method_name / "metrics.json"

    if not results_path.exists():
        if verbose:
            print(f"[{method_name}] results.json not found at {results_path}; skipping")
        return False

    with results_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    per_question = payload.get("per_question", [])
    if not per_question:
        if verbose:
            print(f"[{method_name}] no per_question data found; skipping")
        return False

    metrics = evaluate_results(per_question)
    metrics_payload = {
        "method_name": method_name,
        "top_k": payload.get("top_k", 10),
        **metrics,
    }
    write_json(metrics_payload, metrics_path)

    if verbose:
        overall = metrics.get("overall", {})
        print(
            f"[{method_name}] evaluated n={overall.get('count', 0)} questions — "
            f"Hit@1={overall.get('Hit@1', 0.0):.4f}  "
            f"Hit@10={overall.get('Hit@10', 0.0):.4f}  "
            f"MRR={overall.get('MRR', 0.0):.4f}  "
            f"NDCG={overall.get('NDCG', 0.0):.4f}"
        )
        print(f"  saved -> {metrics_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-stage method(s).")
    parser.add_argument(
        "--method",
        choices=_ALL_CHOICES + ["all"],
        default="all",
        help="Method to evaluate, or 'all' (default). Accepts legacy aliases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RETRIEVAL_OUTPUT_DIR,
        help="Directory containing per-method result subdirs.",
    )
    args = parser.parse_args()

    methods = ALL_METHODS if args.method == "all" else [args.method]
    for method_name in methods:
        _evaluate_method(method_name, args.output_dir, verbose=True)


if __name__ == "__main__":
    main()
