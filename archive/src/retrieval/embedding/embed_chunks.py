import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_texts_and_metadata(records: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts = []
    metadata = []

    for record in records:
        chunk_text = record.get("chunk_text", "").strip()
        if not chunk_text:
            continue

        texts.append(chunk_text)
        metadata.append({
            "paper_id": record.get("paper_id", ""),
            "title": record.get("title", ""),
            "year": record.get("year", ""),
            "authors": record.get("authors", []),
            "tasks": record.get("tasks", []),
            "keywords": record.get("keywords", []),
            "implementations": record.get("implementations", []),
            "chunk_text": chunk_text,
        })

    return texts, metadata


def embed_texts(texts: List[str], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def save_metadata(metadata: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)