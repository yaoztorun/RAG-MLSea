from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.shared.config import resolve_repo_path
from src.pre_retrieval.papers.raw.build_paper_records import build_paper_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical paper records from a local RDF dump.")
    parser.add_argument("--input-path", default="data/raw/pwc_1.nt")
    parser.add_argument("--output", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--stats-output", default="data/intermediate/raw_papers/extraction_stats.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        summary = build_paper_records(
            nt_path=resolve_repo_path(args.input_path),
            output_path=resolve_repo_path(args.output),
            extraction_stats_path=resolve_repo_path(args.stats_output),
            limit=args.limit,
        )
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
