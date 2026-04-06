from __future__ import annotations

import argparse

from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.raw_papers.build_paper_records import build_paper_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical paper records from a local RDF dump.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--nt-path", default="data/raw/pwc_1.nt")
    parser.add_argument("--output", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--predicate-stats", default="data/intermediate/raw_papers/predicate_stats.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_pipeline_config(args.config)
    summary = build_paper_records(
        nt_path=resolve_repo_path(args.nt_path),
        output_path=resolve_repo_path(args.output),
        predicate_stats_path=resolve_repo_path(args.predicate_stats),
        limit=args.limit,
    )
    print(summary)


if __name__ == "__main__":
    main()
