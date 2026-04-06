from __future__ import annotations

import argparse

from src.pre_retrieval.chunking.build_representations import SUPPORTED_REPRESENTATIONS, build_representations
from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper text representations from canonical records.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--records-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--output-dir", default="data/intermediate/representations")
    parser.add_argument("--representation", choices=SUPPORTED_REPRESENTATIONS + ["all"], default="title_only")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.representation == "all":
        representation_types = config["evaluation"]["representation_order"]
    else:
        representation_types = [args.representation]

    counts = build_representations(
        records_path=resolve_repo_path(args.records_path),
        output_dir=resolve_repo_path(args.output_dir),
        representation_types=representation_types,
        representation_config_map=config["representations"],
    )
    print(counts)


if __name__ == "__main__":
    main()
