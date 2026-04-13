from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.shared.config import (
    default_subset_records_path,
    default_subset_stats_path,
    load_pipeline_config,
    resolve_repo_path,
)
from src.pre_retrieval.papers.raw.build_curated_subset import build_curated_subset
from src.pre_retrieval.shared.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a curated local paper subset for fair retrieval experiments.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--papers-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--questions-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--disable-gold-targets", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    subset_config = config.get("corpus_subset", {})
    max_papers = int(args.max_papers or subset_config.get("max_papers", 200000))
    include_gold_targets = bool(subset_config.get("include_gold_targets", True)) and not args.disable_gold_targets

    papers_path = resolve_repo_path(args.papers_path)
    questions_path = resolve_repo_path(args.questions_path or config["evaluation"]["questions_path"])
    output_path = resolve_repo_path(args.output or default_subset_records_path(max_papers))
    stats_output_path = resolve_repo_path(args.stats_output or default_subset_stats_path())

    try:
        require_existing_input(papers_path)
        require_existing_input(questions_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    summary = build_curated_subset(
        papers_master_path=papers_path,
        questions_path=questions_path,
        output_path=output_path,
        stats_output_path=stats_output_path,
        max_papers=max_papers,
        include_gold_targets=include_gold_targets,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
