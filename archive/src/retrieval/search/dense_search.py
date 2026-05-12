from typing import List, Dict, Any

import numpy as np


def cosine_similarity_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = dict(metadata[idx])
        item["score"] = float(scores[idx])
        results.append(item)

    return results