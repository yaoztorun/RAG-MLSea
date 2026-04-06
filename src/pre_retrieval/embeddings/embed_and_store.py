from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pre_retrieval.embeddings.embedder import load_embedder
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.utils import chunked, collection_name_for_representation, load_jsonl, require_existing_input


def _build_store_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "paper_id": record.get("paper_id", ""),
        "title": record.get("title") or "",
        "representation_type": record.get("representation_type", ""),
        "text_length_chars": int(record.get("text_length_chars", 0)),
    }


def embed_and_store_representations(
    representation_path: Path,
    db_path: Path,
    representation_type: str,
    embedder_type: str,
    model_name: str,
    force_rebuild: bool = False,
    batch_size: int = 64,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    require_existing_input(representation_path)
    records = load_jsonl(representation_path)
    if limit is not None:
        records = records[:limit]

    collection_name = collection_name_for_representation(representation_type)
    store = ChromaVectorStore(db_path=db_path, collection_name=collection_name)
    if force_rebuild:
        store.reset()

    ids = [record["item_id"] for record in records]
    existing_ids = set() if force_rebuild else store.get_existing_ids(ids)
    pending_records = [record for record in records if record["item_id"] not in existing_ids]

    if not pending_records:
        return {
            "representation_type": representation_type,
            "collection_name": collection_name,
            "record_count": len(records),
            "inserted_count": 0,
            "skipped_count": len(records),
            "collection_size": store.count(),
        }

    embedder = load_embedder(embedder_type, model_name)
    inserted_count = 0
    for batch in chunked(pending_records, batch_size):
        texts = [record["source_text"] for record in batch]
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        store.upsert(
            ids=[record["item_id"] for record in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_build_store_metadata(record) for record in batch],
        )
        inserted_count += len(batch)

    return {
        "representation_type": representation_type,
        "collection_name": collection_name,
        "record_count": len(records),
        "inserted_count": inserted_count,
        "skipped_count": len(records) - inserted_count,
        "collection_size": store.count(),
    }
