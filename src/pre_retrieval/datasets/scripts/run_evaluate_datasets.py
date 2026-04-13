from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.evaluate_retrieval import evaluate_representation, representation_results_path
from src.pre_retrieval.shared.utils import dataset_collection_name_for_representation, is_dataset_entity_id, require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one stored dataset representation collection.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="dataset_title_only")
    parser.add_argument("--questions-path", default="data/questions/ml_questions_dataset.json")
    parser.add_argument("--records-path", default="data/intermediate/raw_datasets/datasets_master.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    questions_path = resolve_repo_path(args.questions_path)
    records_path = resolve_repo_path(args.records_path)
    output_path = resolve_repo_path(args.output) if args.output else representation_results_path(
        resolve_repo_path(config["evaluation"]["output_dir"]),
        args.representation,
        entity_type="dataset",
    )

    try:
        require_existing_input(questions_path)
        require_existing_input(records_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    collection_name = dataset_collection_name_for_representation(args.representation)
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
        representation_order=config["evaluation"].get("dataset_representation_order", [
            "dataset_title_only",
            "dataset_metadata",
            "dataset_predicate_filtered",
        ]),
        collection_name=collection_name,
        entity_type="dataset",
        id_field="dataset_id",
        target_filter=is_dataset_entity_id,
    )
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
