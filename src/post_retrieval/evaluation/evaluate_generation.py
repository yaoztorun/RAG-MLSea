from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sentence_transformers import SentenceTransformer, util

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
from src.pre_retrieval.utils import save_json

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None


GeneratorFn = Callable[[str, str], str]
BaselineGeneratorFn = Callable[[str], str]


def evaluate_generation(
    *,
    retrieval_results_path: str | Path,
    generator_fn: GeneratorFn,
    baseline_generator_fn: Optional[BaselineGeneratorFn] = None,
    canonical_records_path: str | Path = DEFAULT_CANONICAL_PAPERS_PATH,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    representation_type: Optional[str] = None,
    representations_dir: str | Path = "data/intermediate/representations",
    top_k: int = 3,
    min_retrieval_score: Optional[float] = 0.20,
    rerank_with_cross_encoder: bool = True,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    sas_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit: Optional[int] = None,
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

    sas_model = SentenceTransformer(sas_model_name)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if rouge_scorer is not None else None

    per_question = []
    rag_sas_scores = []
    baseline_sas_scores = []
    rag_rouge_scores = []
    baseline_rouge_scores = []

    entries = get_per_question_entries(retrieval_payload)
    if limit is not None:
        entries = entries[:limit]

    for entry in entries:
        question_id = str(entry.get("question_id", ""))
        question_meta = question_lookup.get(question_id, {})
        question_text = str(entry.get("question") or question_meta.get("question", "")).strip()
        ground_truth = str(question_meta.get("text_answer", "")).strip()
        if not question_text or not ground_truth:
            continue

        context_payload = build_context_payload(
            question_text,
            entry.get("results", []),
            paper_lookup,
            representation_lookup=representation_lookup,
            cross_encoder=cross_encoder,
            use_cross_encoder=rerank_with_cross_encoder,
            min_retrieval_score=min_retrieval_score,
            top_k=top_k,
        )
        rag_answer = generator_fn(question_text, context_payload["context"])
        baseline_answer = baseline_generator_fn(question_text) if baseline_generator_fn is not None else ""

        ground_truth_embedding = sas_model.encode(ground_truth, convert_to_tensor=True)
        rag_embedding = sas_model.encode(rag_answer, convert_to_tensor=True)
        rag_sas = util.cos_sim(ground_truth_embedding, rag_embedding).item()
        rag_sas_scores.append(rag_sas)

        baseline_sas = None
        if baseline_answer:
            baseline_embedding = sas_model.encode(baseline_answer, convert_to_tensor=True)
            baseline_sas = util.cos_sim(ground_truth_embedding, baseline_embedding).item()
            baseline_sas_scores.append(baseline_sas)

        rag_rouge = None
        baseline_rouge = None
        if rouge is not None:
            rag_rouge = rouge.score(ground_truth, rag_answer)["rougeL"].fmeasure
            rag_rouge_scores.append(rag_rouge)
            if baseline_answer:
                baseline_rouge = rouge.score(ground_truth, baseline_answer)["rougeL"].fmeasure
                baseline_rouge_scores.append(baseline_rouge)

        per_question.append(
            {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": ground_truth,
                "context": context_payload["context"],
                "rag_answer": rag_answer,
                "baseline_answer": baseline_answer,
                "metrics": {
                    "rag_sas": rag_sas,
                    "baseline_sas": baseline_sas,
                    "rag_rougeL": rag_rouge,
                    "baseline_rougeL": baseline_rouge,
                },
            }
        )

    payload = {
        "retrieval_results_path": str(retrieval_results_path),
        "representation_type": representation_type or retrieval_payload.get("representation_type"),
        "evaluated_questions": len(per_question),
        "metrics": {
            "rag_sas": (sum(rag_sas_scores) / len(rag_sas_scores)) if rag_sas_scores else 0.0,
            "baseline_sas": (sum(baseline_sas_scores) / len(baseline_sas_scores)) if baseline_sas_scores else None,
            "rag_rougeL": (sum(rag_rouge_scores) / len(rag_rouge_scores)) if rag_rouge_scores else None,
            "baseline_rougeL": (sum(baseline_rouge_scores) / len(baseline_rouge_scores)) if baseline_rouge_scores else None,
        },
        "per_question": per_question,
    }
    if output_path is not None:
        save_json(payload, Path(output_path))
    return payload
