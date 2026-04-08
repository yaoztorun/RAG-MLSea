import json
from pathlib import Path

import numpy as np

from src.retrieval.embedding.embed_queries import embed_query
from src.retrieval.search.dense_search import cosine_similarity_search


EMBEDDINGS_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_embeddings.npy")
METADATA_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_metadata.json")
OUTPUT_RESULTS = Path("data/intermediate/retrieval_results/query_topk_results.json")

QUERY = "Which tasks are covered by the CBLUE benchmark in MLSea-KG?"
TOP_K = 5


def load_metadata(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = load_metadata(METADATA_PATH)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata records: {len(metadata)}")
    print(f"Query: {QUERY}")

    query_vec = embed_query(QUERY)
    results = cosine_similarity_search(
        query_embedding=query_vec,
        doc_embeddings=embeddings,
        metadata=metadata,
        top_k=TOP_K,
    )

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieval results to: {OUTPUT_RESULTS}")
    print("\nTop-k results:")
    for i, item in enumerate(results, start=1):
        print(f"{i}. score={item['score']:.4f} | title={item.get('title', '')}")


if __name__ == "__main__":
    main()