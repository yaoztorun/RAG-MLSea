from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.models.chunking.build_model_representations import (
    SUPPORTED_MODEL_REPRESENTATIONS,
    build_model_representations,
)
from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model text representations from canonical model records.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", default="data/intermediate/raw_models/models_master.jsonl")
    parser.add_argument("--output-dir", default="data/intermediate/representations/models")
    parser.add_argument(
        "--representation",
        choices=SUPPORTED_MODEL_REPRESENTATIONS + ["all"],
        default="model_title_only",
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
        representation_types = SUPPORTED_MODEL_REPRESENTATIONS
    else:
        representation_types = [args.representation]

    model_repr_config = config.get("model_representations", {})
    counts = build_model_representations(
        records_path=input_path,
        output_dir=resolve_repo_path(args.output_dir),
        representation_types=representation_types,
        representation_config_map=model_repr_config,
        limit=args.limit,
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
