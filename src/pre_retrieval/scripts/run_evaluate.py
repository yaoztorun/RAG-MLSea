from __future__ import annotations

import argparse

from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.evaluation.evaluate_retrieval import evaluate_representation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one stored representation collection.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--questions-path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    questions_path = args.questions_path or config["evaluation"]["questions_path"]
    output_path = args.output or f"data/intermediate/retrieval_results/{args.representation}_results.json"

    payload = evaluate_representation(
        representation_type=args.representation,
        questions_path=resolve_repo_path(questions_path),
        db_path=resolve_repo_path(config["vector_store"]["db_path"]),
        model_name=config["embedding_model_name"],
        top_k_values=config["evaluation"]["top_k"],
        output_path=resolve_repo_path(output_path),
    )
    print(payload["summary"])


if __name__ == "__main__":
    main()
