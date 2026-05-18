from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.datasets.chunking.build_dataset_representations import (
    SUPPORTED_DATASET_REPRESENTATIONS,
    build_dataset_representations,
)
from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.gold_target_coverage import check_gold_target_coverage, load_gold_ids_for_entity_type
from src.pre_retrieval.shared.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset text representations from canonical dataset records.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", default="data/intermediate/raw_datasets/datasets_master.jsonl")
    parser.add_argument("--output-dir", default="data/intermediate/representations/datasets")
    parser.add_argument(
        "--representation",
        choices=SUPPORTED_DATASET_REPRESENTATIONS + ["all"],
        default="dataset_title_only",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions-path", default="data/questions/ml_questions_dataset.json")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = resolve_repo_path(args.input_path)
    questions_path = resolve_repo_path(args.questions_path)
    try:
        require_existing_input(input_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    gold_target_ids = None
    if questions_path.exists():
        check_gold_target_coverage(questions_path, input_path, "dataset", "dataset_id")
        gold_target_ids = load_gold_ids_for_entity_type(questions_path, "dataset")
        print(f"[Gold Datasets] Loaded {len(gold_target_ids)} gold dataset IRIs for force-inclusion.", flush=True)

    if args.representation == "all":
        representation_types = SUPPORTED_DATASET_REPRESENTATIONS
    else:
        representation_types = [args.representation]

    dataset_repr_config = config.get("dataset_representations", {})
    counts = build_dataset_representations(
        records_path=input_path,
        output_dir=resolve_repo_path(args.output_dir),
        representation_types=representation_types,
        representation_config_map=dataset_repr_config,
        limit=args.limit,
        gold_target_ids=gold_target_ids,
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
