from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pre_retrieval.config import REPO_ROOT, load_pipeline_config
from src.pre_retrieval.evaluation.aggregate_results import aggregate_result_files
from src.pre_retrieval.embeddings.vector_store import ChromaVectorStore
from src.pre_retrieval.retrieval.retrieve import retrieve_queries
from src.pre_retrieval.utils import (
    build_item_id,
    collection_name_for_representation,
    is_paper_entity_id,
    load_json,
    load_jsonl,
    normalize_identifier,
    save_json,
)


RESULTS_FILE_NAME = "results.json"
TOP10_FILE_NAME = "top10.json"
TOP_DOCUMENT_LIMIT = 10
TOP_DOCUMENT_METADATA_FIELDS = (
    "authors",
    "publication_year",
    "keywords",
    "tasks",
    "datasets",
    "methods",
    "metrics",
    "implementations",
)


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


def representation_results_dir(output_dir: Path, representation_type: str) -> Path:
    return output_dir / representation_type


def representation_results_path(output_dir: Path, representation_type: str) -> Path:
    return representation_results_dir(output_dir, representation_type) / RESULTS_FILE_NAME


def representation_top10_path(output_dir: Path, representation_type: str) -> Path:
    return representation_results_dir(output_dir, representation_type) / TOP10_FILE_NAME


def _load_record_index(records_path: Path) -> Dict[str, Dict[str, Any]]:
    return {normalize_identifier(str(record.get("paper_id", ""))): record for record in load_jsonl(records_path)}


def _top_document_payload(
    question: Dict[str, Any],
    representation_type: str,
    result: Dict[str, Any],
    record: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = {
        "question_id": question.get("id", ""),
        "question": question.get("question", ""),
        "representation_type": representation_type,
        "rank": int(result.get("rank", 0)),
        "item_id": result.get("item_id", ""),
        "paper_id": result.get("paper_id", ""),
        "title": result.get("title", ""),
        "source_text": result.get("source_text", ""),
        "distance": float(result.get("distance", 0.0)),
        "score": float(result.get("score", 0.0)),
    }
    record = record or {}
    for field_name in TOP_DOCUMENT_METADATA_FIELDS:
        payload[field_name] = record.get(field_name, None if field_name == "publication_year" else [])
    return payload


def evaluate_representation(
    representation_type: str,
    questions_path: Path,
    records_path: Path,
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
    answerable_questions_all = [question for question in all_questions if question.get("is_answerable", True) is True]
    paper_target_questions_all: List[Dict[str, Any]] = []
    skipped_non_paper_target_questions: List[Dict[str, Any]] = []
    for question in answerable_questions_all:
        if is_paper_entity_id(str(question.get("target_entity_iri", ""))):
            paper_target_questions_all.append(question)
        else:
            skipped_non_paper_target_questions.append(question)
    answerable_questions = list(paper_target_questions_all)
    if limit is not None:
        answerable_questions = answerable_questions[:limit]

    top_k = max(max(top_k_values), TOP_DOCUMENT_LIMIT)
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
    record_index = _load_record_index(records_path)

    unmatched_targets: List[str] = []
    metric_lists = {f"Hit@{k}": [] for k in top_k_values}
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    per_question: List[Dict[str, Any]] = []
    top10_entries: List[Dict[str, Any]] = []

    for question, results, gold_item_id in zip(answerable_questions, retrieval_results, gold_item_ids):
        gold_paper_id = normalize_identifier(question.get("target_entity_iri", ""))
        ranked_ids = [normalize_identifier(result.get("paper_id", "")) for result in results]
        if gold_item_id not in matched_item_ids:
            unmatched_targets.append(gold_paper_id)

        question_metrics = {f"Hit@{k}": hit_at_k(ranked_ids, gold_paper_id, k) for k in top_k_values}
        question_mrr = reciprocal_rank(ranked_ids, gold_paper_id)
        question_ndcg = ndcg(ranked_ids, gold_paper_id)
        for metric_name, value in question_metrics.items():
            metric_lists[metric_name].append(value)
        mrr_scores.append(question_mrr)
        ndcg_scores.append(question_ndcg)

        top_documents = [
            _top_document_payload(
                question=question,
                representation_type=representation_type,
                result=result,
                record=record_index.get(normalize_identifier(result.get("paper_id", ""))),
            )
            for result in results[:TOP_DOCUMENT_LIMIT]
        ]
        top10_entries.append(
            {
                "question_id": question.get("id", ""),
                "question": question.get("question", ""),
                "gold_paper_id": gold_paper_id,
                "documents": top_documents,
            }
        )
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
        "records_path": str(records_path),
        "embedder": {
            "embedder_type": embedder_type,
            "model_name": model_name,
        },
        "diagnostics": {
            "total_questions": total_questions,
            "answerable_questions": len(answerable_questions_all),
            "paper_target_questions": len(paper_target_questions_all),
            "skipped_non_paper_target_questions": len(skipped_non_paper_target_questions),
            "evaluated_questions": evaluated_count,
            "skipped_questions": total_questions - evaluated_count,
            "unmatched_targets": unmatched_targets,
            "collection_size": store.count(),
        },
        "metrics": metrics,
        "per_question": per_question,
    }
    top10_payload = {
        "representation_type": representation_type,
        "records_path": str(records_path),
        "top_k": TOP_DOCUMENT_LIMIT,
        "entries": top10_entries,
    }
    aggregate_output_dir = output_path.parent.parent if output_path.name == RESULTS_FILE_NAME else output_path.parent
    save_json(payload, output_path)
    save_json(top10_payload, output_path.parent / TOP10_FILE_NAME)
    aggregate_result_files(
        output_dir=aggregate_output_dir,
        representation_order=representation_order or load_pipeline_config()["evaluation"]["representation_order"],
    )
    return payload
