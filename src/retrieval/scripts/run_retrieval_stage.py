"""Run a single retrieval-stage method."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.config import ALL_METHODS, DEFAULT_TOP_K, METHOD_ALIASES, RETRIEVAL_OUTPUT_DIR
from src.retrieval.pipeline import run_method

_ALL_CHOICES = sorted(set(ALL_METHODS) | set(METHOD_ALIASES.keys()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retrieval-stage method.")
    parser.add_argument(
        "--method",
        required=True,
        choices=_ALL_CHOICES,
        help="Retrieval method to run (new or legacy alias name).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of candidates to produce (default: %(default)s).",
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
        help="Rerun even if output already exists.",
    )
    args = parser.parse_args()

    run_method(
        method_name=args.method,
        top_k=args.top_k,
        output_dir=args.output_dir,
        force=args.force,
        verbose=True,
    )


if __name__ == "__main__":
    main()
