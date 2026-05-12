from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.shared.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.shared.embed_and_store import embed_and_store_representations
from src.pre_retrieval.shared.utils import dataset_collection_name_for_representation


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed one dataset representation file and persist it in Chroma.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="dataset_title_only")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = args.input_path or f"data/intermediate/representations/datasets/{args.representation}.jsonl"
    collection_name = dataset_collection_name_for_representation(args.representation)
    try:
        summary = embed_and_store_representations(
            representation_path=resolve_repo_path(input_path),
            vector_store_config=config["vector_store"],
            representation_type=args.representation,
            embedder_type=config["embedder_type"],
            model_name=config["model_name"],
            force_rebuild=args.force_rebuild,
            batch_size=args.batch_size,
            limit=args.limit,
            collection_name=collection_name,
        )
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
