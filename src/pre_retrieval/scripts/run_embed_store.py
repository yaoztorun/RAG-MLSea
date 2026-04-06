from __future__ import annotations

import argparse

from src.pre_retrieval.config import load_pipeline_config, resolve_repo_path
from src.pre_retrieval.embeddings.embed_and_store import embed_and_store_representations


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed one representation file and persist it in Chroma.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--input", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    representation_path = args.input or f"data/intermediate/representations/{args.representation}.jsonl"
    summary = embed_and_store_representations(
        representation_path=resolve_repo_path(representation_path),
        db_path=resolve_repo_path(config["vector_store"]["db_path"]),
        collection_name=args.representation,
        model_name=config["embedding_model_name"],
        force_rebuild=args.force_rebuild,
        batch_size=args.batch_size,
    )
    print(summary)


if __name__ == "__main__":
    main()
