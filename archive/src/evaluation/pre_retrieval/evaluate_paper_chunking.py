from pathlib import Path
import json
from typing import Dict, List, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.evaluation.utils.loaders import load_json, load_jsonl
from src.evaluation.utils.metrics import hit_at_k, ndcg, reciprocal_rank
from src.evaluation.utils.normalize import normalize_target_iri, normalize_chunk_paper_id
from src.evaluation.utils.reporting import print_results_table
from src.retrieval.embedding.embed_chunks import (
    extract_texts_and_metadata,
    save_embeddings,
    save_metadata,
)


# ---------------- CONFIG ---------------- #

QUESTIONS_PATH = Path("data/questions/ml_questions_dataset.json")

STRATEGY_CONFIGS = {
    "basic": {
        "chunk_path": Path("data/intermediate/chunks/papers/papers_basic_sample.jsonl"),
        "embeddings_path": Path("data/intermediate/embeddings/papers_basic_sample_embeddings.npy"),
        "metadata_path": Path("data/intermediate/embeddings/papers_basic_sample_metadata.json"),
    },
    "enriched": {
        "chunk_path": Path("data/intermediate/chunks/papers/papers_enriched_sample.jsonl"),
        "embeddings_path": Path("data/intermediate/embeddings/papers_enriched_sample_embeddings.npy"),
        "metadata_path": Path("data/intermediate/embeddings/papers_enriched_sample_metadata.json"),
    },
    "one_hop": {
        "chunk_path": Path("data/intermediate/chunks/papers/papers_one_hop_sample.jsonl"),
        "embeddings_path": Path("data/intermediate/embeddings/papers_one_hop_sample_embeddings.npy"),
        "metadata_path": Path("data/intermediate/embeddings/papers_one_hop_sample_metadata.json"),
    },
    "predicate_filtered": {
        "chunk_path": Path("data/intermediate/chunks/papers/papers_predicate_filtered_sample.jsonl"),
        "embeddings_path": Path("data/intermediate/embeddings/papers_predicate_filtered_sample_embeddings.npy"),
        "metadata_path": Path("data/intermediate/embeddings/papers_predicate_filtered_sample_metadata.json"),
    },
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- HELPERS ---------------- #

def filter_paper_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        q for q in questions
        if q.get("question_type", "").startswith("paper_") and q.get("is_answerable", True) is True
    ]


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list metadata in {path}, got {type(payload)}")
    return payload


def ensure_embedding_cache(
    strategy_name: str,
    config: Dict[str, Path],
    model: SentenceTransformer,
) -> tuple[np.ndarray, List[str]]:
    chunk_path = config["chunk_path"]
    embeddings_path = config["embeddings_path"]
    metadata_path = config["metadata_path"]

    if embeddings_path.exists() and metadata_path.exists():
        print(f"Using cached embeddings for strategy: {strategy_name}")
        embeddings = np.load(embeddings_path, mmap_mode="r")
        metadata = load_metadata(metadata_path)
    else:
        print(f"Cache not found for strategy: {strategy_name}. Building from {chunk_path}")
        chunk_records = load_jsonl(chunk_path)
        texts, metadata = extract_texts_and_metadata(chunk_records)

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        save_embeddings(embeddings, embeddings_path)
        save_metadata(metadata, metadata_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embeddings/metadata size mismatch for {strategy_name}: "
            f"{embeddings.shape[0]} != {len(metadata)}"
        )

    chunk_ids = [
        normalize_chunk_paper_id(record.get("paper_id", ""))
        for record in metadata
    ]

    return embeddings, chunk_ids


# ---------------- CORE EVALUATION ---------------- #

def evaluate_strategy(
    strategy_name: str,
    chunk_embeddings: np.ndarray,
    chunk_ids: List[str],
    question_embeddings: np.ndarray,
    gold_ids: List[str],
) -> Dict[str, float]:
    hit1_scores, hit5_scores, mrr_scores, ndcg_scores = [], [], [], []

    for idx, gold_id in enumerate(gold_ids):
        query_embedding = question_embeddings[idx]
        scores = np.dot(chunk_embeddings, query_embedding)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [chunk_ids[i] for i in ranked_indices]

        hit1_scores.append(hit_at_k(ranked_ids, gold_id, 1))
        hit5_scores.append(hit_at_k(ranked_ids, gold_id, 5))
        mrr_scores.append(reciprocal_rank(ranked_ids, gold_id))
        ndcg_scores.append(ndcg(ranked_ids, gold_id))

    n = len(gold_ids)

    return {
        "hit@1": sum(hit1_scores) / n if n else 0.0,
        "hit@5": sum(hit5_scores) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
        "ndcg": sum(ndcg_scores) / n if n else 0.0,
    }


# ---------------- MAIN ---------------- #

def main() -> None:
    print("Loading questions...")
    questions = load_json(QUESTIONS_PATH)
    paper_questions = filter_paper_questions(questions)

    print(f"Total questions loaded: {len(questions)}")
    print(f"Paper questions selected: {len(paper_questions)}")

    # ---- DEBUG NORMALIZATION CHECK ---- #
    print("\n--- DEBUG: IRI NORMALIZATION ---")
    if paper_questions:
        sample_q = paper_questions[0]
        print("Raw target_entity_iri:")
        print(sample_q["target_entity_iri"])
        print("Normalized target:")
        print(normalize_target_iri(sample_q["target_entity_iri"]))
    print("--------------------------------\n")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    question_texts = [q.get("question", "") for q in paper_questions]
    gold_ids = [
        normalize_target_iri(q.get("target_entity_iri", ""))
        for q in paper_questions
    ]

    print("Encoding all questions once...")
    question_embeddings = model.encode(
        question_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    results: Dict[str, Dict[str, float]] = {}

    for strategy_name, config in STRATEGY_CONFIGS.items():
        print(f"\nPreparing retrieval data for strategy: {strategy_name}")
        chunk_embeddings, chunk_ids = ensure_embedding_cache(strategy_name, config, model)
        print(f"Chunk count: {len(chunk_ids)}")

        metrics = evaluate_strategy(
            strategy_name,
            chunk_embeddings,
            chunk_ids,
            question_embeddings,
            gold_ids,
        )

        results[strategy_name] = metrics

    print_results_table(results)


if __name__ == "__main__":
    main()