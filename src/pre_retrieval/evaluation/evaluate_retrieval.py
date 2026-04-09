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
METRIC_NAMES = ("MRR", "NDCG")


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


def _clean_segment_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _is_unanswerable_question(question: Dict[str, Any]) -> bool:
    return question.get("is_answerable", True) is False or _clean_segment_value(question.get("target_entity_iri")) is None


def _is_answerable_question(question: Dict[str, Any]) -> bool:
    return not _is_unanswerable_question(question)


def _build_metric_container(top_k_values: Sequence[int]) -> Dict[str, List[float]]:
    return {f"Hit@{k}": [] for k in top_k_values} | {"MRR": [], "NDCG": []}


def _new_segment_state(top_k_values: Sequence[int]) -> Dict[str, Any]:
    return {
        "total_questions": 0,
        "answerable_questions": 0,
        "paper_target_questions": 0,
        "evaluated_questions": 0,
        "skipped_questions": 0,
        "skipped_non_paper_targets": 0,
        "skipped_unanswerable": 0,
        "metrics": _build_metric_container(top_k_values),
    }


def _finalize_metric_payload(metric_lists: Dict[str, List[float]], evaluated_count: int) -> Dict[str, float]:
    return {
        metric_name: (sum(values) / evaluated_count if evaluated_count else 0.0)
        for metric_name, values in metric_lists.items()
    }


def _segment_output(segment_state: Dict[str, Any], top_k_values: Sequence[int]) -> Dict[str, Any]:
    metric_names = [f"Hit@{k}" for k in top_k_values] + list(METRIC_NAMES)
    payload = {
        "total_questions": segment_state["total_questions"],
        "answerable_questions": segment_state["answerable_questions"],
        "paper_target_questions": segment_state["paper_target_questions"],
        "evaluated_questions": segment_state["evaluated_questions"],
        "skipped_questions": segment_state["total_questions"] - segment_state["evaluated_questions"],
        "skipped_non_paper_targets": segment_state["skipped_non_paper_targets"],
        "skipped_unanswerable": segment_state["skipped_unanswerable"],
    }
    payload.update(_finalize_metric_payload(segment_state["metrics"], segment_state["evaluated_questions"]))
    for metric_name in metric_names:
        payload.setdefault(metric_name, 0.0)
    return payload


def _segment_counts(segment_output: Dict[str, Any], top_k_values: Sequence[int]) -> Dict[str, int]:
    metric_names = {f"Hit@{k}" for k in top_k_values} | set(METRIC_NAMES)
    return {key: value for key, value in segment_output.items() if key not in metric_names}


def _unanswerable_diagnostics(total_unanswerable_questions: int) -> Dict[str, int]:
    """Keep both legacy/general skip keys and the explicit unanswerable alias in sync."""
    return {
        "total_unanswerable_questions": total_unanswerable_questions,
        "skipped_unanswerable": total_unanswerable_questions,
        "unanswerable_questions_skipped": total_unanswerable_questions,
    }


def _build_segment_maps(questions: Sequence[Dict[str, Any]], top_k_values: Sequence[int], field_name: str) -> Dict[str, Dict[str, Any]]:
    segment_values = []
    for question in questions:
        value = _clean_segment_value(question.get(field_name))
        if value is not None and value not in segment_values:
            segment_values.append(value)
    return {value: _new_segment_state(top_k_values) for value in segment_values}


def _update_segment_counts(segment_maps: Dict[str, Dict[str, Any]], question: Dict[str, Any], field_name: str) -> None:
    segment_value = _clean_segment_value(question.get(field_name))
    if segment_value is None or segment_value not in segment_maps:
        return

    segment_state = segment_maps[segment_value]
    segment_state["total_questions"] += 1
    if _is_answerable_question(question):
        segment_state["answerable_questions"] += 1
        if is_paper_entity_id(str(question.get("target_entity_iri", ""))):
            segment_state["paper_target_questions"] += 1
        else:
            segment_state["skipped_non_paper_targets"] += 1
    else:
        segment_state["skipped_unanswerable"] += 1


def _update_segment_metrics(
    segment_maps: Dict[str, Dict[str, Any]],
    question: Dict[str, Any],
    field_name: str,
    question_metrics: Dict[str, float],
) -> None:
    segment_value = _clean_segment_value(question.get(field_name))
    if segment_value is None or segment_value not in segment_maps:
        return

    segment_state = segment_maps[segment_value]
    segment_state["evaluated_questions"] += 1
    for metric_name, metric_value in question_metrics.items():
        segment_state["metrics"][metric_name].append(metric_value)


def build_evaluation_payload(
    *,
    representation_type: str,
    collection_name: str,
    records_path: Path,
    embedder_type: str,
    model_name: str,
    top_k_values: Sequence[int],
    all_questions: Sequence[Dict[str, Any]],
    evaluated_questions: Sequence[Dict[str, Any]],
    retrieval_results: Sequence[Sequence[Dict[str, Any]]],
    matched_item_ids: Sequence[str],
    collection_size: int,
    record_index: Dict[str, Dict[str, Any]],
    abstention_score_threshold: float | None = None,
    unanswerable_results: Sequence[Sequence[Dict[str, Any]]] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    difficulty_segments = _build_segment_maps(all_questions, top_k_values, "difficulty")
    category_segments = _build_segment_maps(all_questions, top_k_values, "category")
    for question in all_questions:
        _update_segment_counts(difficulty_segments, question, "difficulty")
        _update_segment_counts(category_segments, question, "category")

    answerable_questions_all = [question for question in all_questions if _is_answerable_question(question)]
    unanswerable_questions = [question for question in all_questions if _is_unanswerable_question(question)]
    paper_target_questions_all: List[Dict[str, Any]] = []
    skipped_non_paper_target_questions: List[Dict[str, Any]] = []
    for question in answerable_questions_all:
        if is_paper_entity_id(str(question.get("target_entity_iri", ""))):
            paper_target_questions_all.append(question)
        else:
            skipped_non_paper_target_questions.append(question)

    matched_item_ids_set = set(matched_item_ids)
    unmatched_targets: List[str] = []
    metric_lists = _build_metric_container(top_k_values)
    per_question: List[Dict[str, Any]] = []
    top10_entries: List[Dict[str, Any]] = []

    for question, results in zip(evaluated_questions, retrieval_results):
        gold_paper_id = normalize_identifier(question.get("target_entity_iri", ""))
        ranked_ids = [normalize_identifier(result.get("paper_id", "")) for result in results]
        gold_item_id = build_item_id(representation_type, gold_paper_id)
        if gold_item_id not in matched_item_ids_set:
            unmatched_targets.append(gold_paper_id)

        question_metrics = {f"Hit@{k}": hit_at_k(ranked_ids, gold_paper_id, k) for k in top_k_values}
        question_metrics["MRR"] = reciprocal_rank(ranked_ids, gold_paper_id)
        question_metrics["NDCG"] = ndcg(ranked_ids, gold_paper_id)
        for metric_name, value in question_metrics.items():
            metric_lists[metric_name].append(value)

        _update_segment_metrics(difficulty_segments, question, "difficulty", question_metrics)
        _update_segment_metrics(category_segments, question, "category", question_metrics)

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
                "difficulty": _clean_segment_value(question.get("difficulty")),
                "category": _clean_segment_value(question.get("category")),
                "is_answerable": question.get("is_answerable", True),
                "target_entity_iri": question.get("target_entity_iri", ""),
                "gold_paper_id": gold_paper_id,
                "matched_in_collection": gold_item_id in matched_item_ids_set,
                "metrics": question_metrics,
                "results": results,
            }
        )

    evaluated_count = len(per_question)
    metrics = _finalize_metric_payload(metric_lists, evaluated_count)
    metrics_by_difficulty = {
        difficulty: _segment_output(segment_state, top_k_values)
        for difficulty, segment_state in difficulty_segments.items()
    }
    metrics_by_category = {
        category: _segment_output(segment_state, top_k_values)
        for category, segment_state in category_segments.items()
    }

    diagnostics = {
        "total_questions": len(all_questions),
        "answerable_questions": len(answerable_questions_all),
        "paper_target_questions": len(paper_target_questions_all),
        "skipped_non_paper_target_questions": len(skipped_non_paper_target_questions),
        "skipped_non_paper_targets": len(skipped_non_paper_target_questions),
        **_unanswerable_diagnostics(len(unanswerable_questions)),
        "evaluated_questions": evaluated_count,
        "skipped_questions": len(all_questions) - evaluated_count,
        "unmatched_targets": unmatched_targets,
        "collection_size": collection_size,
        "counts_by_difficulty": {
            difficulty: _segment_counts(segment_output, top_k_values)
            for difficulty, segment_output in metrics_by_difficulty.items()
        },
        "counts_by_category": {
            category: _segment_counts(segment_output, top_k_values)
            for category, segment_output in metrics_by_category.items()
        },
    }

    if abstention_score_threshold is not None:
        unanswerable_rows = list(zip(unanswerable_questions, unanswerable_results or []))
        abstained_count = sum(
            1
            for _, results in unanswerable_rows
            if not results or float(results[0].get("score", 0.0)) < abstention_score_threshold
        )
        false_accept_count = len(unanswerable_rows) - abstained_count
        abstention_payload = {
            "score_threshold": abstention_score_threshold,
            "evaluated_unanswerable_questions": len(unanswerable_rows),
            "total_unanswerable_questions": len(unanswerable_questions),
            "unanswerable_questions_skipped": max(len(unanswerable_questions) - len(unanswerable_rows), 0),
            "abstained_unanswerable_questions": abstained_count,
            "false_accept_count": false_accept_count,
            "unanswerable_rejection_rate": (abstained_count / len(unanswerable_rows) if unanswerable_rows else 0.0),
            "false_accept_rate": (false_accept_count / len(unanswerable_rows) if unanswerable_rows else 0.0),
        }
        diagnostics["abstention"] = abstention_payload
        if "unanswerable" in metrics_by_category:
            metrics_by_category["unanswerable"] = metrics_by_category["unanswerable"] | abstention_payload

    payload = {
        "representation_type": representation_type,
        "collection_name": collection_name,
        "records_path": str(records_path),
        "embedder": {
            "embedder_type": embedder_type,
            "model_name": model_name,
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
        "metrics_by_difficulty": metrics_by_difficulty,
        "metrics_by_category": metrics_by_category,
        "per_question": per_question,
    }
    top10_payload = {
        "representation_type": representation_type,
        "records_path": str(records_path),
        "top_k": TOP_DOCUMENT_LIMIT,
        "entries": top10_entries,
    }
    return payload, top10_payload


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
    abstention_score_threshold: float | None = None,
) -> Dict[str, Any]:
    all_questions = load_json(questions_path)
    answerable_questions_all = [question for question in all_questions if _is_answerable_question(question)]
    paper_target_questions_all: List[Dict[str, Any]] = []
    for question in answerable_questions_all:
        if is_paper_entity_id(str(question.get("target_entity_iri", ""))):
            paper_target_questions_all.append(question)
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
    unanswerable_questions = [question for question in all_questions if _is_unanswerable_question(question)]
    unanswerable_results = None
    if abstention_score_threshold is not None and unanswerable_questions:
        unanswerable_results = retrieve_queries(
            queries=[question.get("question", "") for question in unanswerable_questions],
            vector_store_config=vector_store_config,
            representation_type=representation_type,
            embedder_type=embedder_type,
            model_name=model_name,
            top_k=1,
        )

    payload, top10_payload = build_evaluation_payload(
        representation_type=representation_type,
        collection_name=collection_name_for_representation(representation_type),
        records_path=records_path,
        embedder_type=embedder_type,
        model_name=model_name,
        top_k_values=top_k_values,
        all_questions=all_questions,
        evaluated_questions=answerable_questions,
        retrieval_results=retrieval_results,
        matched_item_ids=matched_item_ids,
        collection_size=store.count(),
        record_index=record_index,
        abstention_score_threshold=abstention_score_threshold,
        unanswerable_results=unanswerable_results,
    )
    aggregate_output_dir = output_path.parent.parent if output_path.name == RESULTS_FILE_NAME else output_path.parent
    save_json(payload, output_path)
    save_json(top10_payload, output_path.parent / TOP10_FILE_NAME)
    aggregate_result_files(
        output_dir=aggregate_output_dir,
        representation_order=representation_order or load_pipeline_config()["evaluation"]["representation_order"],
    )
    return payload
