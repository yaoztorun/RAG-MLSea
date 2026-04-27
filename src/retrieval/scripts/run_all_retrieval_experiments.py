"""Run all five retrieval-stage methods and aggregate summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.config import DEFAULT_TOP_K, RETRIEVAL_OUTPUT_DIR
from src.retrieval.pipeline import run_all_methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all retrieval-stage methods and aggregate summaries."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of candidates to produce per method (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RETRIEVAL_OUTPUT_DIR,
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun methods even if outputs already exist.",
    )
    args = parser.parse_args()

    run_all_methods(
        top_k=args.top_k,
        output_dir=args.output_dir,
        force=args.force,
        verbose=True,
    )


if __name__ == "__main__":
    main()
