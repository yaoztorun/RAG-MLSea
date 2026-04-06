from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.pre_retrieval.embeddings.embedder import load_embedder
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.utils import collection_name_for_representation


def _parse_query_result(query_result: Dict[str, Any], top_k: int) -> List[List[Dict[str, Any]]]:
    results: List[List[Dict[str, Any]]] = []
    for ids, distances, metadatas, documents in zip(
        query_result.get("ids", []),
        query_result.get("distances", []),
        query_result.get("metadatas", []),
        query_result.get("documents", []),
    ):
        query_rows: List[Dict[str, Any]] = []
        for rank, (doc_id, distance, metadata, document) in enumerate(
            zip(ids[:top_k], distances[:top_k], metadatas[:top_k], documents[:top_k]),
            start=1,
        ):
            metadata = metadata or {}
            query_rows.append(
                {
                    "rank": rank,
                    "item_id": doc_id,
                    "paper_id": metadata.get("paper_id", ""),
                    "title": metadata.get("title", ""),
                    "representation_type": metadata.get("representation_type", ""),
                    "text_length_chars": metadata.get("text_length_chars", 0),
                    "distance": float(distance),
                    "score": float(1.0 - distance),
                    "source_text": document,
                }
            )
        results.append(query_rows)
    return results


def retrieve_queries(
    queries: Iterable[str],
    db_path: Path,
    representation_type: str,
    embedder_type: str,
    model_name: str,
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    query_list = list(queries)
    if not query_list:
        return []

    embedder = load_embedder(embedder_type, model_name)
    query_embeddings = embedder.encode(
        query_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    store = ChromaVectorStore(db_path=db_path, collection_name=collection_name_for_representation(representation_type))
    query_result = store.query(query_embeddings=query_embeddings, n_results=top_k)
    return _parse_query_result(query_result, top_k)
