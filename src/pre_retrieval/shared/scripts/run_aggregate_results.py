from __future__ import annotations

import argparse

from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.aggregate_results import aggregate_result_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate existing retrieval result files into shared summary outputs.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    results_dir = resolve_repo_path(args.results_dir or config["evaluation"]["output_dir"])
    summary = aggregate_result_files(
        output_dir=results_dir,
        representation_order=config["evaluation"]["representation_order"],
    )
    print(f"Aggregated {len(summary['rows'])} result rows into {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
