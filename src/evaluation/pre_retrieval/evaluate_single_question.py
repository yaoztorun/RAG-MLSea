from pathlib import Path
from typing import Dict, List, Any
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from src.evaluation.utils.loaders import load_json
from src.evaluation.utils.metrics import hit_at_k, ndcg, reciprocal_rank
from src.evaluation.utils.normalize import normalize_target_iri, normalize_chunk_paper_id


# CONFIG
QUESTIONS_PATH = Path(os.getenv("MLSEA_QUESTIONS_PATH", "data/questions/ml_questions_dataset.json"))
EMBEDDINGS_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_embeddings.npy")
METADATA_PATH = Path("data/intermediate/embeddings/papers_enriched_sample_metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_ID = os.getenv("MLSEA_QUESTION_ID", "").strip()
QUESTION_SCENARIO = os.getenv("MLSEA_QUESTION_SCENARIO", "paper_answerable").strip().lower()
TOP_K = int(os.getenv("MLSEA_TOP_K", "15"))
INCLUDE_UNANSWERABLE_EXAMPLE = os.getenv("MLSEA_INCLUDE_UNANSWERABLE_EXAMPLE", "1").strip() in {"1", "true", "True"}
ANSWERABLE_EXAMPLE_COUNT = int(os.getenv("MLSEA_ANSWERABLE_EXAMPLE_COUNT", "3"))
DECIMALS = 6


def filter_paper_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        q for q in questions
        if q.get("question_type", "").startswith("paper_")
    ]


def select_questions_by_scenario(questions: List[Dict[str, Any]], scenario: str) -> List[Dict[str, Any]]:
    paper_questions = filter_paper_questions(questions)

    if scenario == "paper_all":
        return paper_questions
    if scenario == "paper_answerable":
        return [q for q in paper_questions if q.get("is_answerable", True) is True]
    if scenario == "all_unanswerable":
        return [q for q in questions if q.get("is_answerable", True) is False]
    if scenario == "all":
        return questions

    raise ValueError(
        "Unsupported MLSEA_QUESTION_SCENARIO. Use one of: "
        "paper_answerable, paper_all, all_unanswerable, all"
    )


def select_single_question(questions: List[Dict[str, Any]], question_id: str) -> Dict[str, Any]:
    if question_id:
        for question in questions:
            if question.get("id") == question_id:
                return question
        print(f"WARNING: Question id not found in selected scenario: {question_id}. Using first question in scenario.")
    return questions[0]


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list metadata in {path}, got {type(payload)}")
    return payload


def evaluate_one_question(
    question: Dict[str, Any],
    chunk_embeddings: np.ndarray,
    chunk_ids: List[str],
    chunk_papers: List[str],
    model: SentenceTransformer,
    top_k: int,
) -> Dict[str, Any]:
    q_id = question.get("id", "unknown")
    q_text = question.get("question", "")
    q_type = question.get("question_type", "")
    is_answerable = question.get("is_answerable", True)
    answer_type = question.get("answer_type", "")
    gold_iri = question.get("target_entity_iri", "")
    gold_id = normalize_target_iri(gold_iri)

    print(f"\n{'='*100}")
    print(f"QUESTION ID: {q_id}")
    print(f"QUESTION TYPE: {q_type}")
    print(f"IS ANSWERABLE: {is_answerable}")
    print(f"ANSWER TYPE: {answer_type}")
    print(f"QUESTION: {q_text}")
    print(f"GOLD IRI (raw): {gold_iri}")
    print(f"GOLD ID (normalized): {gold_id}")
    print(f"{'='*100}")

    query_embedding = model.encode(
        q_text,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    scores = np.dot(chunk_embeddings, query_embedding)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_ids = [chunk_ids[i] for i in ranked_indices]
    ranked_scores = scores[ranked_indices]
    ranked_papers = [chunk_papers[i] for i in ranked_indices]

    top1_score = float(ranked_scores[0]) if len(ranked_scores) > 0 else 0.0
    top2_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else 0.0
    top1_margin = top1_score - top2_score

    print("\nRETRIEVAL CONFIDENCE SUMMARY:")
    print(f"  Top1 score:   {top1_score:.{DECIMALS}f}")
    print(f"  Top2 score:   {top2_score:.{DECIMALS}f}")
    print(f"  Top1 margin:  {top1_margin:.{DECIMALS}f}")

    if is_answerable and gold_id:
        hit1 = hit_at_k(ranked_ids, gold_id, 1)
        hit5 = hit_at_k(ranked_ids, gold_id, 5)
        mrr = reciprocal_rank(ranked_ids, gold_id)
        ndcg_score = ndcg(ranked_ids, gold_id)

        gold_rank = ranked_ids.index(gold_id) + 1 if gold_id in ranked_ids else None
        gold_score = float(scores[ranked_indices[gold_rank - 1]]) if gold_rank else None
        gap_to_top1 = (top1_score - gold_score) if gold_score is not None else None

        print(f"\n{'='*100}")
        print("METRICS (ANSWERABLE QUESTION):")
        print(f"  Hit@1:      {hit1:.{DECIMALS}f}")
        print(f"  Hit@5:      {hit5:.{DECIMALS}f}")
        print(f"  MRR:        {mrr:.{DECIMALS}f}")
        print(f"  NDCG:       {ndcg_score:.{DECIMALS}f}")
        print(f"  Gold rank:  {gold_rank if gold_rank is not None else 'NOT_FOUND'}")
        if gold_score is not None:
            print(f"  Gold score: {gold_score:.{DECIMALS}f}")
            print(f"  Gap top1-gold: {gap_to_top1:.{DECIMALS}f}")
        print(f"{'='*100}\n")
        summary = {
            "id": q_id,
            "type": q_type,
            "answerable": True,
            "hit@1": hit1,
            "hit@5": hit5,
            "mrr": mrr,
            "ndcg": ndcg_score,
            "gold_rank": gold_rank,
            "top1": top1_score,
            "top2": top2_score,
            "margin": top1_margin,
        }
    else:
        abstain_recommended = (top1_score < 0.30) or (top1_margin < 0.02)
        print(f"\n{'='*100}")
        print("UNANSWERABLE ANALYSIS:")
        print("  Ranking metrics are skipped because no gold answer exists.")
        print(f"  Abstain recommendation: {'YES' if abstain_recommended else 'NO'}")
        print("  Rule: abstain if Top1 < 0.30 OR Top1 margin < 0.02")
        print(f"{'='*100}\n")
        summary = {
            "id": q_id,
            "type": q_type,
            "answerable": False,
            "hit@1": None,
            "hit@5": None,
            "mrr": None,
            "ndcg": None,
            "gold_rank": None,
            "top1": top1_score,
            "top2": top2_score,
            "margin": top1_margin,
        }

    print(f"TOP {top_k} RETRIEVED PAPERS:\n")
    print(f"{'Rank':<6} {'Score':<12} {'MarginVsTop1':<14} {'GoldMatch':<10} {'Paper ID':<55}")
    print(f"{'-'*110}")

    limit = min(top_k, len(ranked_ids))
    for rank in range(limit):
        score = float(ranked_scores[rank])
        margin_vs_top1 = top1_score - score
        is_gold = "YES" if ranked_ids[rank] == gold_id and bool(gold_id) else "NO"
        ranked_paper = ranked_papers[rank][:52] + "..." if len(ranked_papers[rank]) > 55 else ranked_papers[rank]
        print(
            f"{rank+1:<6} {score:>11.{DECIMALS}f} "
            f"{margin_vs_top1:>13.{DECIMALS}f} "
            f"{is_gold:<10} {ranked_paper:<55}"
        )

    return summary


def build_question_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("QUESTION COMPARISON SUMMARY")
    print("=" * 100)
    header = (
        f"{'QID':<12} {'Type':<26} {'Ans':<5} {'Hit@1':>8} {'Hit@5':>8} "
        f"{'MRR':>10} {'NDCG':>10} {'GoldRank':>9} {'Top1':>10} {'Margin':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        def fm(v: Any) -> str:
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.{DECIMALS}f}"
            return str(v)

        print(
            f"{str(row['id'])[:12]:<12} "
            f"{str(row['type'])[:26]:<26} "
            f"{('Y' if row['answerable'] else 'N'):<5} "
            f"{fm(row['hit@1']):>8} "
            f"{fm(row['hit@5']):>8} "
            f"{fm(row['mrr']):>10} "
            f"{fm(row['ndcg']):>10} "
            f"{fm(row['gold_rank']):>9} "
            f"{fm(row['top1']):>10} "
            f"{fm(row['margin']):>10}"
        )


def main() -> None:
    print("Loading questions...")
    questions = load_json(QUESTIONS_PATH)
    selected_questions = select_questions_by_scenario(questions, QUESTION_SCENARIO)

    print(f"Total questions: {len(questions)}")
    print(f"Scenario: {QUESTION_SCENARIO}")
    print(f"Selected questions in scenario: {len(selected_questions)}")

    if not selected_questions:
        print("No questions found for the selected scenario!")
        return

    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    print(f"Loading metadata from: {METADATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    chunk_embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    metadata = load_metadata(METADATA_PATH)

    print(f"Embeddings shape: {chunk_embeddings.shape}")
    print(f"Metadata records: {len(metadata)}")
    print("Search scope: all rows in loaded metadata")

    if len(metadata) == 0:
        print("No metadata records found!")
        return

    if chunk_embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embeddings row count and metadata length do not match: "
            f"{chunk_embeddings.shape[0]} != {len(metadata)}"
        )

    chunk_ids = [
        normalize_chunk_paper_id(record.get("paper_id", ""))
        for record in metadata
    ]
    chunk_papers = [record.get("paper_id", "") for record in metadata]

    # Load model once, then evaluate one or more demo questions.
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    primary_question = select_single_question(selected_questions, QUESTION_ID)
    demo_questions: List[Dict[str, Any]] = []

    answerable_in_selected = [q for q in selected_questions if q.get("is_answerable", True) is True]

    if primary_question.get("is_answerable", True) is True:
        demo_questions.append(primary_question)

    for q in answerable_in_selected:
        if len([x for x in demo_questions if x.get("is_answerable", True) is True]) >= ANSWERABLE_EXAMPLE_COUNT:
            break
        if q.get("id") not in {x.get("id") for x in demo_questions}:
            demo_questions.append(q)

    if not demo_questions:
        demo_questions.append(primary_question)

    if INCLUDE_UNANSWERABLE_EXAMPLE:
        opposite = [
            q for q in questions
            if q.get("is_answerable", True) is not primary_question.get("is_answerable", True)
        ]
        if opposite:
            first_opposite = opposite[0]
            if first_opposite.get("id") != primary_question.get("id"):
                demo_questions.append(first_opposite)

    print(f"\nRunning detailed output for {len(demo_questions)} question(s)...")

    summaries: List[Dict[str, Any]] = []

    for question in demo_questions:
        summary = evaluate_one_question(
            question=question,
            chunk_embeddings=chunk_embeddings,
            chunk_ids=chunk_ids,
            chunk_papers=chunk_papers,
            model=model,
            top_k=TOP_K,
        )
        summaries.append(summary)

    build_question_summary_table(summaries)


if __name__ == "__main__":
    main()
