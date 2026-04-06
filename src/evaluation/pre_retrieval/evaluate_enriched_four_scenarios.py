from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.evaluation.utils.loaders import load_json
from src.evaluation.utils.metrics import hit_at_k, ndcg, reciprocal_rank
from src.evaluation.utils.normalize import normalize_chunk_paper_id, normalize_target_iri


QUESTIONS_PATH = Path("data/questions/ml_questions_dataset.json")
CHUNK_PATH = Path("data/intermediate/chunks/papers/papers_enriched_sample.jsonl")
EMBEDDINGS_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_embeddings.npy")
METADATA_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOW_PARTIAL = os.getenv("MLSEA_ALLOW_PARTIAL_METADATA", "0").strip() in {"1", "true", "True"}

SCENARIOS = [
    "paper_answerable",
    "paper_all",
    "all_unanswerable",
    "all",
]

DECIMALS = 6


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list metadata in {path}, got {type(payload)}")
    return payload


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def select_questions_by_scenario(questions: List[Dict[str, Any]], scenario: str) -> List[Dict[str, Any]]:
    paper_questions = [
        q for q in questions
        if str(q.get("question_type", "")).startswith("paper_")
    ]

    if scenario == "paper_answerable":
        return [q for q in paper_questions if q.get("is_answerable", True) is True]
    if scenario == "paper_all":
        return paper_questions
    if scenario == "all_unanswerable":
        return [q for q in questions if q.get("is_answerable", True) is False]
    if scenario == "all":
        return questions

    raise ValueError(f"Unsupported scenario: {scenario}")


def compact_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.{DECIMALS}f}"


def summarize(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "max": float(np.max(arr)),
    }


def build_table(results: List[Dict[str, Any]]) -> None:
    headers = (
        f"{'Scenario':<18} {'Q':>5} {'Ans':>5} {'Unans':>7} "
        f"{'EvalN':>6} {'H1#':>5} {'H5#':>5} "
        f"{'Hit@1':>10} {'Hit@5':>10} {'MRR':>10} {'NDCG':>10} "
        f"{'Top1':>10} {'Margin':>10} {'LowConf':>10}"
    )
    print("\nScenario comparison (enriched embeddings)\n")
    print(headers)
    print("-" * len(headers))

    for row in results:
        print(
            f"{row['scenario']:<18} "
            f"{row['q_count']:>5} "
            f"{row['answerable_count']:>5} "
            f"{row['unanswerable_count']:>7} "
            f"{row['eval_count']:>6} "
            f"{row['hit@1_count']:>5} "
            f"{row['hit@5_count']:>5} "
            f"{compact_metric(row['hit@1']):>10} "
            f"{compact_metric(row['hit@5']):>10} "
            f"{compact_metric(row['mrr']):>10} "
            f"{compact_metric(row['ndcg']):>10} "
            f"{compact_metric(row['avg_top1_score']):>10} "
            f"{compact_metric(row['avg_top1_margin']):>10} "
            f"{compact_metric(row['low_conf_rate']):>10}"
        )


def print_detailed_breakdown(results: List[Dict[str, Any]]) -> None:
    print("\nDetailed scenario breakdown\n")
    for row in results:
        print(f"[{row['scenario']}]")
        print(
            f"  Questions: total={row['q_count']}, answerable={row['answerable_count']}, "
            f"unanswerable={row['unanswerable_count']}, eval_n={row['eval_count']}"
        )
        print(
            f"  Hits: hit@1_count={row['hit@1_count']}, hit@5_count={row['hit@5_count']}, "
            f"gold_found_rate={compact_metric(row['gold_found_rate'])}"
        )
        print(
            f"  Metrics: hit@1={compact_metric(row['hit@1'])}, hit@5={compact_metric(row['hit@5'])}, "
            f"mrr={compact_metric(row['mrr'])}, ndcg={compact_metric(row['ndcg'])}, "
            f"avg_gold_rank={compact_metric(row['avg_gold_rank'])}"
        )

        top1_stats = row["top1_stats"]
        margin_stats = row["margin_stats"]
        print(
            f"  Top1 score stats: mean={compact_metric(top1_stats['mean'])}, "
            f"std={compact_metric(top1_stats['std'])}, min={compact_metric(top1_stats['min'])}, "
            f"p50={compact_metric(top1_stats['p50'])}, max={compact_metric(top1_stats['max'])}"
        )
        print(
            f"  Margin stats: mean={compact_metric(margin_stats['mean'])}, "
            f"std={compact_metric(margin_stats['std'])}, min={compact_metric(margin_stats['min'])}, "
            f"p50={compact_metric(margin_stats['p50'])}, max={compact_metric(margin_stats['max'])}"
        )
        print(f"  Low confidence rate: {compact_metric(row['low_conf_rate'])}")
        print()


def evaluate_scenario(
    scenario: str,
    questions: List[Dict[str, Any]],
    model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    chunk_ids: List[str],
    low_conf_threshold: float,
) -> Dict[str, Any]:
    selected = select_questions_by_scenario(questions, scenario)

    if not selected:
        return {
            "scenario": scenario,
            "q_count": 0,
            "answerable_count": 0,
            "unanswerable_count": 0,
            "hit@1": None,
            "hit@5": None,
            "mrr": None,
            "ndcg": None,
            "avg_top1_score": None,
            "avg_top1_margin": None,
            "low_conf_rate": None,
        }

    texts = [q.get("question", "") for q in selected]
    q_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    answerable = [q for q in selected if q.get("is_answerable", True) is True]
    unanswerable = [q for q in selected if q.get("is_answerable", True) is False]

    hit1_scores: List[float] = []
    hit5_scores: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    gold_ranks: List[float] = []

    top1_scores: List[float] = []
    top1_margins: List[float] = []
    low_conf_flags: List[float] = []

    for idx, question in enumerate(selected):
        scores = np.dot(chunk_embeddings, q_embeddings[idx])
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [chunk_ids[i] for i in ranked_indices]

        top1 = float(scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0
        top2 = float(scores[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
        margin = top1 - top2

        top1_scores.append(top1)
        top1_margins.append(margin)
        low_conf_flags.append(1.0 if top1 < low_conf_threshold else 0.0)

        if question.get("is_answerable", True) is not True:
            continue

        gold_id = normalize_target_iri(question.get("target_entity_iri", ""))
        if not gold_id:
            continue

        if gold_id in ranked_ids:
            gold_ranks.append(float(ranked_ids.index(gold_id) + 1))

        hit1_scores.append(hit_at_k(ranked_ids, gold_id, 1))
        hit5_scores.append(hit_at_k(ranked_ids, gold_id, 5))
        mrr_scores.append(reciprocal_rank(ranked_ids, gold_id))
        ndcg_scores.append(ndcg(ranked_ids, gold_id))

    eval_count = len(hit1_scores)
    hit1_count = int(sum(hit1_scores))
    hit5_count = int(sum(hit5_scores))
    gold_found_rate = (len(gold_ranks) / eval_count) if eval_count else None

    return {
        "scenario": scenario,
        "q_count": len(selected),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "eval_count": eval_count,
        "hit@1_count": hit1_count,
        "hit@5_count": hit5_count,
        "hit@1": (sum(hit1_scores) / len(hit1_scores)) if hit1_scores else None,
        "hit@5": (sum(hit5_scores) / len(hit5_scores)) if hit5_scores else None,
        "mrr": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else None,
        "ndcg": (sum(ndcg_scores) / len(ndcg_scores)) if ndcg_scores else None,
        "avg_gold_rank": (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None,
        "gold_found_rate": gold_found_rate,
        "avg_top1_score": sum(top1_scores) / len(top1_scores),
        "avg_top1_margin": sum(top1_margins) / len(top1_margins),
        "low_conf_rate": sum(low_conf_flags) / len(low_conf_flags),
        "top1_stats": summarize(top1_scores),
        "margin_stats": summarize(top1_margins),
    }


def main() -> None:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Missing enriched embedding cache. Run: "
            "python -m src.retrieval.scripts.run_embed_enriched_chunks"
        )

    if not CHUNK_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_PATH}")

    print(f"Loading questions from: {QUESTIONS_PATH}")
    questions = load_json(QUESTIONS_PATH)
    print(f"Questions loaded: {len(questions)}")

    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    chunk_embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    print(f"Loading metadata from: {METADATA_PATH}")
    metadata = load_metadata(METADATA_PATH)

    if chunk_embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embeddings and metadata size mismatch: "
            f"{chunk_embeddings.shape[0]} != {len(metadata)}"
        )

    chunk_rows = count_jsonl_rows(CHUNK_PATH)
    metadata_rows = len(metadata)
    print(f"Enriched chunk rows: {chunk_rows}")
    print(f"Metadata rows: {metadata_rows}")

    if metadata_rows != chunk_rows:
        if not ALLOW_PARTIAL:
            raise RuntimeError(
                "Metadata does not cover all enriched chunk rows. "
                "Regenerate full embeddings with: "
                "python -m src.retrieval.scripts.run_embed_enriched_chunks"
            )
        print("WARNING: Partial metadata mode is enabled for demo purposes.")
        print("WARNING: Metrics are computed on subset embeddings, not full enriched corpus.")

    chunk_ids = [normalize_chunk_paper_id(row.get("paper_id", "")) for row in metadata]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    paper_answerable = select_questions_by_scenario(questions, "paper_answerable")
    if paper_answerable:
        paper_texts = [q.get("question", "") for q in paper_answerable]
        paper_q_embeddings = model.encode(
            paper_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        answerable_top1: List[float] = []
        for idx in range(len(paper_answerable)):
            scores = np.dot(chunk_embeddings, paper_q_embeddings[idx])
            ranked_indices = np.argsort(scores)[::-1]
            answerable_top1.append(float(scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0)
        low_conf_threshold = float(np.percentile(answerable_top1, 10))
    else:
        low_conf_threshold = 0.25

    print(f"Low-confidence threshold (Top1 score): {low_conf_threshold:.{DECIMALS}f}")

    results = [
        evaluate_scenario(
            scenario=scenario,
            questions=questions,
            model=model,
            chunk_embeddings=chunk_embeddings,
            chunk_ids=chunk_ids,
            low_conf_threshold=low_conf_threshold,
        )
        for scenario in SCENARIOS
    ]

    build_table(results)
    print_detailed_breakdown(results)

    print("\nUnanswerable interpretation")
    print("- Retrieval always returns nearest papers, even when no true answer exists.")
    print("- Use Top1 score and Top1-Top2 margin to detect weak-confidence results.")
    print("- LowConf is the fraction of queries with Top1 below the threshold.")


if __name__ == "__main__":
    main()
