from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.pre_retrieval.embeddings.embedder import load_embedder
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.io_utils import chunked, load_jsonl


def _build_store_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    field_stats = record.get("field_stats", {})
    return {
        "paper_id": record.get("paper_id", ""),
        "title": record.get("title", ""),
        "representation_type": record.get("representation_type", ""),
        "source_text": record.get("source_text", ""),
        "text_length": int(record.get("text_length", 0)),
        "year": record.get("year", ""),
        "author_count": int(field_stats.get("author_count", 0)),
        "task_count": int(field_stats.get("task_count", 0)),
        "dataset_count": int(field_stats.get("dataset_count", 0)),
        "method_count": int(field_stats.get("method_count", 0)),
        "metric_count": int(field_stats.get("metric_count", 0)),
        "implementation_count": int(field_stats.get("implementation_count", 0)),
        "keyword_count": int(field_stats.get("keyword_count", 0)),
        "linked_entity_count": int(field_stats.get("linked_entity_count", 0)),
    }


def embed_and_store_representations(
    representation_path: Path,
    db_path: Path,
    collection_name: str,
    model_name: str,
    force_rebuild: bool = False,
    batch_size: int = 64,
) -> Dict[str, Any]:
    records = load_jsonl(representation_path)
    store = ChromaVectorStore(db_path=db_path, collection_name=collection_name)

    if force_rebuild:
        store.reset()

    ids = [record["id"] for record in records]
    existing_ids = set() if force_rebuild else store.get_existing_ids(ids)
    pending_records = [record for record in records if force_rebuild or record["id"] not in existing_ids]

    if not pending_records:
        return {
            "collection_name": collection_name,
            "record_count": len(records),
            "inserted_count": 0,
            "skipped_count": len(records),
            "collection_size": store.count(),
        }

    model = load_embedder(model_name)
    inserted_count = 0

    for batch in chunked(pending_records, batch_size):
        texts = [record["source_text"] for record in batch]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        store.upsert(
            ids=[record["id"] for record in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_build_store_metadata(record) for record in batch],
        )
        inserted_count += len(batch)

    return {
        "collection_name": collection_name,
        "record_count": len(records),
        "inserted_count": inserted_count,
        "skipped_count": len(records) - inserted_count,
        "collection_size": store.count(),
    }
