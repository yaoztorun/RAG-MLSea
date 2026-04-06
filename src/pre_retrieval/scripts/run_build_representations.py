from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.chunking.build_representations import SUPPORTED_REPRESENTATIONS, build_representations
from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper text representations from canonical records.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--output-dir", default="data/intermediate/representations")
    parser.add_argument("--representation", choices=SUPPORTED_REPRESENTATIONS + ["all"], default="title_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = resolve_repo_path(args.input_path)
    try:
        require_existing_input(input_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.representation == "all":
        representation_types = config["evaluation"]["representation_order"]
    else:
        representation_types = [args.representation]

    counts = build_representations(
        records_path=input_path,
        output_dir=resolve_repo_path(args.output_dir),
        representation_types=representation_types,
        representation_config_map=config["representations"],
        limit=args.limit,
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
