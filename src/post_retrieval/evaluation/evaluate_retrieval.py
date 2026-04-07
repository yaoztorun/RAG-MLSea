from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.post_retrieval.pipeline.context_builder import build_context_payload
from src.post_retrieval.pipeline.data_loading import (
    DEFAULT_CANONICAL_PAPERS_PATH,
    DEFAULT_QUESTIONS_PATH,
    build_paper_id_lookup,
    build_question_id_lookup,
    build_representation_lookup,
    get_per_question_entries,
    load_canonical_paper_records,
    load_question_dataset,
    load_representation_records,
    load_retrieval_payload,
)
from src.post_retrieval.pipeline.post_retrieval_pipeline import DEFAULT_CROSS_ENCODER_MODEL, load_cross_encoder
from src.pre_retrieval.utils import normalize_identifier, save_json


def evaluate_retrieval_results(
    *,
    retrieval_results_path: str | Path,
    canonical_records_path: str | Path = DEFAULT_CANONICAL_PAPERS_PATH,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    representation_type: Optional[str] = None,
    representations_dir: str | Path = "data/intermediate/representations",
    top_k: int = 3,
    rerank_with_cross_encoder: bool = True,
    min_retrieval_score: Optional[float] = 0.20,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    retrieval_payload = load_retrieval_payload(retrieval_results_path)
    paper_lookup = build_paper_id_lookup(load_canonical_paper_records(canonical_records_path))
    question_lookup = build_question_id_lookup(load_question_dataset(questions_path))
    representation_lookup = None
    if representation_type:
        representation_lookup = build_representation_lookup(
            load_representation_records(representation_type, representations_dir=representations_dir)
        )

    cross_encoder = None
    if rerank_with_cross_encoder:
        cross_encoder = load_cross_encoder(cross_encoder_model_name)

    per_question = []
    baseline_successes = 0
    reranked_successes = 0
    evaluated_questions = 0

    for entry in get_per_question_entries(retrieval_payload):
        question_id = str(entry.get("question_id", ""))
        question_meta = question_lookup.get(question_id, {})
        if question_meta:
            if question_meta.get("is_answerable", True) is not True:
                continue
            if not str(question_meta.get("question_type", "")).startswith("paper_"):
                continue

        question_text = str(entry.get("question") or question_meta.get("question", "")).strip()
        gold_paper_id = normalize_identifier(entry.get("gold_paper_id") or question_meta.get("target_entity_iri", ""))
        results = list(entry.get("results", []))
        if not question_text or not gold_paper_id or not results:
            continue

        baseline_ids = [normalize_identifier(result.get("paper_id", "")) for result in results[:top_k]]
        reranked_payload = build_context_payload(
            question_text,
            results,
            paper_lookup,
            representation_lookup=representation_lookup,
            cross_encoder=cross_encoder,
            use_cross_encoder=rerank_with_cross_encoder,
            min_retrieval_score=min_retrieval_score,
            top_k=top_k,
        )
        reranked_ids = [
            normalize_identifier(result.get("paper_id", ""))
            for result in reranked_payload.get("selected_results", [])
        ]

        baseline_hit = gold_paper_id in baseline_ids
        reranked_hit = gold_paper_id in reranked_ids
        baseline_successes += int(baseline_hit)
        reranked_successes += int(reranked_hit)
        evaluated_questions += 1
        per_question.append(
            {
                "question_id": question_id,
                "question": question_text,
                "gold_paper_id": gold_paper_id,
                "baseline_top_k": baseline_ids,
                "reranked_top_k": reranked_ids,
                "baseline_hit": baseline_hit,
                "reranked_hit": reranked_hit,
            }
        )

    payload = {
        "retrieval_results_path": str(retrieval_results_path),
        "representation_type": representation_type or retrieval_payload.get("representation_type"),
        "top_k": top_k,
        "evaluated_questions": evaluated_questions,
        "metrics": {
            f"baseline_recall@{top_k}": (baseline_successes / evaluated_questions) if evaluated_questions else 0.0,
            f"reranked_recall@{top_k}": (reranked_successes / evaluated_questions) if evaluated_questions else 0.0,
        },
        "per_question": per_question,
    }
    if output_path is not None:
        save_json(payload, Path(output_path))
    return payload
