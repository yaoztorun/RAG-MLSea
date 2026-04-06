from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence, Set

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from src.pre_retrieval.utils import chunked


class VectorStore(ABC):
    @abstractmethod
    def get_existing_ids(self, ids: Sequence[str]) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embeddings: Sequence[Sequence[float]], n_results: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, db_path: Path, collection_name: str) -> None:
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection_name = collection_name
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_existing_ids(self, ids: Sequence[str]) -> Set[str]:
        existing: Set[str] = set()
        if not ids:
            return existing
        for batch in chunked(list(ids), 500):
            result = self._collection.get(ids=list(batch), include=[])
            existing.update(result.get("ids", []))
        return existing

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=np.asarray(embeddings, dtype=float).tolist(),
            metadatas=list(metadatas),
        )

    def query(self, query_embeddings: Sequence[Sequence[float]], n_results: int) -> Dict[str, Any]:
        return self._collection.query(
            query_embeddings=np.asarray(query_embeddings, dtype=float).tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except ValueError:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
