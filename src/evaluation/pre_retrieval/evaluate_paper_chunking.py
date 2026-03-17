from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.evaluation.utils.loaders import load_json, load_jsonl
from src.evaluation.utils.metrics import hit_at_k, reciprocal_rank
from src.evaluation.utils.normalize import normalize_target_iri, normalize_chunk_paper_id
from src.evaluation.utils.reporting import print_results_table


QUESTIONS_PATH = Path("data/questions/ml_questions.json")

CHUNK_FILES = {
    "basic": Path("data/intermediate/chunks/papers/papers_basic_sample.jsonl"),
    "enriched": Path("data/intermediate/chunks/papers/papers_enriched_sample.jsonl"),
    "one_hop": Path("data/intermediate/chunks/papers/papers_one_hop_sample.jsonl"),
    "predicate_filtered": Path("data/intermediate/chunks/papers/papers_predicate_filtered_sample.jsonl"),
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def filter_paper_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for q in questions:
        qtype = q.get("question_type", "")
        if qtype.startswith("paper_"):
            filtered.append(q)
    return filtered


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    if query_norm == 0:
        return np.zeros(len(doc_matrix), dtype=float)

    safe_doc_norms = np.where(doc_norms == 0, 1e-12, doc_norms)
    return np.dot(doc_matrix, query_vec) / (safe_doc_norms * query_norm)


def evaluate_strategy(
    strategy_name: str,
    chunk_records: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> Dict[str, float]:
    chunk_texts = [record.get("chunk_text", "") for record in chunk_records]
    chunk_ids = [normalize_chunk_paper_id(record.get("paper_id", "")) for record in chunk_records]

    print(f"\nEncoding chunks for strategy: {strategy_name}")
    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )

    hit1_scores: List[float] = []
    hit5_scores: List[float] = []
    hit10_scores: List[float] = []
    mrr_scores: List[float] = []

    for q in questions:
        question_text = q.get("question", "")
        gold_id = normalize_target_iri(q.get("target_entity_iri", ""))

        query_embedding = model.encode(
            question_text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        scores = cosine_similarity_matrix(query_embedding, chunk_embeddings)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [chunk_ids[i] for i in ranked_indices]

        hit1_scores.append(hit_at_k(ranked_ids, gold_id, 1))
        hit5_scores.append(hit_at_k(ranked_ids, gold_id, 5))
        hit10_scores.append(hit_at_k(ranked_ids, gold_id, 10))
        mrr_scores.append(reciprocal_rank(ranked_ids, gold_id))

    n = len(questions)
    return {
        "hit@1": sum(hit1_scores) / n if n else 0.0,
        "hit@5": sum(hit5_scores) / n if n else 0.0,
        "hit@10": sum(hit10_scores) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
    }


def main() -> None:
    print("Loading questions...")
    questions = load_json(QUESTIONS_PATH)
    paper_questions = filter_paper_questions(questions)

    print(f"Total questions loaded: {len(questions)}")
    print(f"Paper questions selected: {len(paper_questions)}")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    results: Dict[str, Dict[str, float]] = {}

    for strategy_name, chunk_path in CHUNK_FILES.items():
        print(f"\nLoading chunks for strategy '{strategy_name}' from {chunk_path}")
        chunk_records = load_jsonl(chunk_path)
        print(f"Chunk count: {len(chunk_records)}")

        metrics = evaluate_strategy(strategy_name, chunk_records, paper_questions, model)
        results[strategy_name] = metrics

    print_results_table(results)


if __name__ == "__main__":
    main()