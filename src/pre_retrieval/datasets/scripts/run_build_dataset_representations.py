from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.datasets.chunking.build_dataset_representations import (
    SUPPORTED_DATASET_REPRESENTATIONS,
    build_dataset_representations,
)
from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
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
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = resolve_repo_path(args.input_path)
    try:
        require_existing_input(input_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

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
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
