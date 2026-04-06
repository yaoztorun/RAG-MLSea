from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"\w+")


@dataclass
class HashingEmbedder:
    dimensions: int = 768

    def encode(
        self,
        texts: Iterable[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        rows: List[str] = [str(text or "") for text in texts]
        embeddings = np.zeros((len(rows), self.dimensions), dtype=np.float32)

        for row_index, text in enumerate(rows):
            for token in TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % self.dimensions
                embeddings[row_index, bucket] += 1.0

        if normalize_embeddings and len(rows) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms

        return embeddings if convert_to_numpy else embeddings.tolist()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Iterable[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self._model.encode(
            list(texts),
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


def load_embedder(model_name: str) -> SentenceTransformerEmbedder | HashingEmbedder:
    if model_name.startswith("hashing://"):
        suffix = model_name.split("://", 1)[1].strip()
        dimensions = int(suffix) if suffix else 768
        return HashingEmbedder(dimensions=dimensions)
    if model_name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
