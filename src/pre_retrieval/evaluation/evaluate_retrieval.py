from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.pre_retrieval.embeddings.embedder import load_embedder
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.io_utils import load_json, normalize_identifier, save_json
from src.pre_retrieval.retrieval.retrieve import _parse_query_result


def filter_paper_questions(questions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        question
        for question in questions
        if question.get("question_type", "").startswith("paper_")
        and question.get("is_answerable", True) is True
    ]


def reciprocal_rank(ranked_ids: Sequence[str], gold_id: str) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            return 1.0 / index
    return 0.0


def ndcg(ranked_ids: Sequence[str], gold_id: str) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            from math import log2

            return 1.0 / log2(index + 1)
    return 0.0


def hit_at_k(ranked_ids: Sequence[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def evaluate_representation(
    representation_type: str,
    questions_path: Path,
    db_path: Path,
    model_name: str,
    top_k_values: Sequence[int],
    output_path: Path,
) -> Dict[str, Any]:
    questions = filter_paper_questions(load_json(questions_path))
    if not questions:
        raise ValueError(f"No answerable paper questions found in {questions_path}")

    model = load_embedder(model_name)
    query_embeddings = model.encode(
        [question.get("question", "") for question in questions],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    max_top_k = max(top_k_values)
    store = ChromaVectorStore(db_path=db_path, collection_name=representation_type)
    raw_results = store.query(query_embeddings=query_embeddings, n_results=max_top_k)
    ranked_results = _parse_query_result(raw_results, max_top_k)

    metrics = {f"top_{k}": [] for k in top_k_values}
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for question, retrieved_rows in zip(questions, ranked_results):
        gold_id = normalize_identifier(question.get("target_entity_iri", ""))
        ranked_ids = [normalize_identifier(row["paper_id"]) for row in retrieved_rows]
        question_metrics = {f"top_{k}": hit_at_k(ranked_ids, gold_id, k) for k in top_k_values}
        question_mrr = reciprocal_rank(ranked_ids, gold_id)
        question_ndcg = ndcg(ranked_ids, gold_id)

        for key, value in question_metrics.items():
            metrics[key].append(value)
        mrr_scores.append(question_mrr)
        ndcg_scores.append(question_ndcg)

        per_question.append(
            {
                "question_id": question.get("id", ""),
                "question": question.get("question", ""),
                "gold_paper_id": gold_id,
                "metrics": question_metrics | {"mrr": question_mrr, "ndcg": question_ndcg},
                "results": retrieved_rows,
            }
        )

    question_count = len(per_question)
    summary = {
        key: (sum(values) / question_count if question_count else 0.0)
        for key, values in metrics.items()
    }
    summary["mrr"] = sum(mrr_scores) / question_count if question_count else 0.0
    summary["ndcg"] = sum(ndcg_scores) / question_count if question_count else 0.0

    payload = {
        "representation_type": representation_type,
        "model_name": model_name,
        "question_count": question_count,
        "summary": summary,
        "per_question": per_question,
    }
    save_json(payload, output_path)
    return payload
