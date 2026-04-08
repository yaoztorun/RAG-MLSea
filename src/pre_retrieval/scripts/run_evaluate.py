from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.config import load_pipeline_config, resolve_records_path, resolve_repo_path
from src.pre_retrieval.evaluation.evaluate_retrieval import evaluate_representation, representation_results_path
from src.pre_retrieval.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one stored representation collection.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--records-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--disable-subset", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = args.input_path or config["evaluation"]["questions_path"]
    questions_path = resolve_repo_path(input_path)
    records_path = resolve_records_path(
        config,
        args.records_path,
        disable_subset=args.disable_subset,
        max_papers=args.max_papers,
    )
    output_path = resolve_repo_path(args.output) if args.output else representation_results_path(
        resolve_repo_path(config["evaluation"]["output_dir"]),
        args.representation,
    )

    try:
        require_existing_input(questions_path)
        require_existing_input(records_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = evaluate_representation(
        representation_type=args.representation,
        questions_path=questions_path,
        records_path=records_path,
        vector_store_config=config["vector_store"],
        embedder_type=config["embedder_type"],
        model_name=config["model_name"],
        top_k_values=config["evaluation"]["top_k"],
        output_path=output_path,
        limit=args.limit,
        representation_order=config["evaluation"]["representation_order"],
    )
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
