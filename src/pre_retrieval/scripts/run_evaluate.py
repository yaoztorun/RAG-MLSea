from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.evaluation.evaluate_retrieval import evaluate_representation
from src.pre_retrieval.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one stored representation collection.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = args.input_path or config["evaluation"]["questions_path"]
    questions_path = resolve_repo_path(input_path)
    output_path = resolve_repo_path(args.output or f"{config['evaluation']['output_dir']}/{args.representation}_results.json")

    try:
        require_existing_input(questions_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = evaluate_representation(
        representation_type=args.representation,
        questions_path=questions_path,
        db_path=resolve_repo_path(config["vector_store"]["db_path"]),
        embedder_type=config["embedder_type"],
        model_name=config["model_name"],
        top_k_values=config["evaluation"]["top_k"],
        output_path=output_path,
        limit=args.limit,
    )
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
