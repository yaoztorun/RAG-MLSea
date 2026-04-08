import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.embedding.embed_chunks import (
    MODEL_NAME,
    save_embeddings,
    save_metadata,
)


INPUT_JSONL = Path("data/intermediate/chunks/papers/papers_enriched_sample.jsonl")
OUTPUT_EMBEDDINGS = Path("data/intermediate/embeddings/papers_enriched_sample_embeddings.npy")
OUTPUT_METADATA = Path("data/intermediate/embeddings/papers_enriched_sample_metadata.json")
BATCH_SIZE = int(os.getenv("MLSEA_EMBED_BATCH_SIZE", "512"))


def iter_texts_and_metadata(path: Path, batch_size: int) -> Iterator[Tuple[List[str], List[Dict[str, Any]]]]:
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            chunk_text = str(record.get("chunk_text", "")).strip()
            if not chunk_text:
                continue

            texts.append(chunk_text)
            # Keep metadata compact for full-corpus runs.
            metadata.append(
                {
                    "paper_id": record.get("paper_id", ""),
                    "title": record.get("title", ""),
                    "year": record.get("year", ""),
                }
            )

            if len(texts) >= batch_size:
                yield texts, metadata
                texts = []
                metadata = []

    if texts:
        yield texts, metadata


def main() -> None:
    print(f"Loading chunks from: {INPUT_JSONL}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    model = SentenceTransformer(MODEL_NAME)

    all_embeddings: List[np.ndarray] = []
    all_metadata: List[Dict[str, Any]] = []
    total_texts = 0
    batch_no = 0

    for texts, metadata in iter_texts_and_metadata(INPUT_JSONL, BATCH_SIZE):
        batch_no += 1
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)
        total_texts += len(texts)

        if batch_no % 20 == 0:
            print(f"Processed batches: {batch_no} | embedded texts: {total_texts}")

    if not all_embeddings:
        raise RuntimeError("No valid chunk_text rows found to embed.")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Metadata rows: {len(all_metadata)}")

    save_embeddings(embeddings, OUTPUT_EMBEDDINGS)
    save_metadata(all_metadata, OUTPUT_METADATA)

    print(f"Saved embeddings to: {OUTPUT_EMBEDDINGS}")
    print(f"Saved metadata to: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
