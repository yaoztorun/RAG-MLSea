from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

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


@dataclass(frozen=True)
class ChromaConnectionConfig:
    chroma_mode: str = "http"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    persist_directory: str = "data/intermediate/chroma"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ChromaConnectionConfig":
        return cls(
            chroma_mode=str(config.get("chroma_mode", "http")).strip().lower() or "http",
            chroma_host=str(config.get("chroma_host", "localhost")).strip() or "localhost",
            chroma_port=int(config.get("chroma_port", 8000)),
            persist_directory=str(config.get("persist_directory") or config.get("db_path") or "data/intermediate/chroma"),
        )


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        chroma_mode: str = "http",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        persist_directory: Optional[Path] = None,
    ) -> None:
        if chroma_mode == "persistent":
            if persist_directory is None:
                raise ValueError("persist_directory is required when chroma_mode='persistent'.")
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        elif chroma_mode == "http":
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            raise ValueError(f"Unsupported chroma_mode: {chroma_mode}")
        self._collection_name = collection_name
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_config(cls, collection_name: str, vector_store_config: Dict[str, Any], repo_root: Path) -> "ChromaVectorStore":
        config = ChromaConnectionConfig.from_dict(vector_store_config)
        persist_directory = repo_root / config.persist_directory if not Path(config.persist_directory).is_absolute() else Path(config.persist_directory)
        return cls(
            collection_name=collection_name,
            chroma_mode=config.chroma_mode,
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
            persist_directory=persist_directory,
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
