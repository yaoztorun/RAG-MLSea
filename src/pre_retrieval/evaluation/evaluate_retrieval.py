from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pre_retrieval.config import REPO_ROOT, load_pipeline_config
from src.pre_retrieval.evaluation.aggregate_results import aggregate_result_files
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.retrieval.retrieve import retrieve_queries
from src.pre_retrieval.utils import build_item_id, collection_name_for_representation, load_json, normalize_identifier, save_json


def hit_at_k(ranked_ids: Sequence[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_id: str) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            return 1.0 / index
    return 0.0


def ndcg(ranked_ids: Sequence[str], gold_id: str) -> float:
    from math import log2

    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            return 1.0 / log2(index + 1)
    return 0.0


def evaluate_representation(
    representation_type: str,
    questions_path: Path,
    vector_store_config: Dict[str, Any],
    embedder_type: str,
    model_name: str,
    top_k_values: Sequence[int],
    output_path: Path,
    limit: Optional[int] = None,
    representation_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    all_questions = load_json(questions_path)
    total_questions = len(all_questions)
    paper_questions = [question for question in all_questions if question.get("question_type", "").startswith("paper_")]
    answerable_questions_all = [question for question in paper_questions if question.get("is_answerable", True) is True]
    answerable_questions = list(answerable_questions_all)
    if limit is not None:
        answerable_questions = answerable_questions[:limit]

    top_k = max(top_k_values)
    retrieval_results = retrieve_queries(
        queries=[question.get("question", "") for question in answerable_questions],
        vector_store_config=vector_store_config,
        representation_type=representation_type,
        embedder_type=embedder_type,
        model_name=model_name,
        top_k=top_k,
    )

    store = ChromaVectorStore.from_config(
        collection_name=collection_name_for_representation(representation_type),
        vector_store_config=vector_store_config,
        repo_root=REPO_ROOT,
    )
    gold_item_ids = [build_item_id(representation_type, normalize_identifier(question.get("target_entity_iri", ""))) for question in answerable_questions]
    matched_item_ids = store.get_existing_ids(gold_item_ids)

    unmatched_targets: List[str] = []
    metric_lists = {f"Hit@{k}": [] for k in top_k_values}
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for question, results, gold_item_id in zip(answerable_questions, retrieval_results, gold_item_ids):
        gold_paper_id = normalize_identifier(question.get("target_entity_iri", ""))
        ranked_ids = [normalize_identifier(result.get("paper_id", "")) for result in results]
        if gold_item_id not in matched_item_ids:
            unmatched_targets.append(gold_paper_id)

        question_metrics = {
            f"Hit@{k}": hit_at_k(ranked_ids, gold_paper_id, k)
            for k in top_k_values
        }
        question_mrr = reciprocal_rank(ranked_ids, gold_paper_id)
        question_ndcg = ndcg(ranked_ids, gold_paper_id)
        for metric_name, value in question_metrics.items():
            metric_lists[metric_name].append(value)
        mrr_scores.append(question_mrr)
        ndcg_scores.append(question_ndcg)
        per_question.append(
            {
                "question_id": question.get("id", ""),
                "question": question.get("question", ""),
                "target_entity_iri": question.get("target_entity_iri", ""),
                "gold_paper_id": gold_paper_id,
                "matched_in_collection": gold_item_id in matched_item_ids,
                "metrics": question_metrics | {"MRR": question_mrr, "NDCG": question_ndcg},
                "results": results,
            }
        )

    evaluated_count = len(per_question)
    metrics = {
        metric_name: (sum(values) / evaluated_count if evaluated_count else 0.0)
        for metric_name, values in metric_lists.items()
    }
    metrics["MRR"] = sum(mrr_scores) / evaluated_count if evaluated_count else 0.0
    metrics["NDCG"] = sum(ndcg_scores) / evaluated_count if evaluated_count else 0.0

    payload = {
        "representation_type": representation_type,
        "collection_name": collection_name_for_representation(representation_type),
        "embedder": {
            "embedder_type": embedder_type,
            "model_name": model_name,
        },
        "diagnostics": {
            "total_questions": total_questions,
            "answerable_questions": len(answerable_questions_all),
            "evaluated_questions": evaluated_count,
            "skipped_questions": total_questions - evaluated_count,
            "unmatched_targets": unmatched_targets,
            "collection_size": store.count(),
        },
        "metrics": metrics,
        "per_question": per_question,
    }
    save_json(payload, output_path)
    aggregate_result_files(
        output_dir=output_path.parent,
        representation_order=representation_order or load_pipeline_config()["evaluation"]["representation_order"],
    )
    return payload
