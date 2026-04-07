from __future__ import annotations

import argparse
import json

from src.post_retrieval.evaluation import evaluate_retrieval_results
from src.post_retrieval.pipeline import resolve_retrieval_results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline retrieval results before and after post-retrieval reranking.")
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--retrieval-results-path")
    parser.add_argument("--papers-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--questions-path", default="data/questions/ml_questions_dataset.json")
    parser.add_argument("--representations-dir", default="data/intermediate/representations")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    parser.add_argument("--output-path", default="data/intermediate/post_retrieval/retrieval_evaluation.json")
    args = parser.parse_args()

    retrieval_results_path = args.retrieval_results_path or str(resolve_retrieval_results_path(args.representation))
    payload = evaluate_retrieval_results(
        retrieval_results_path=retrieval_results_path,
        canonical_records_path=args.papers_path,
        questions_path=args.questions_path,
        representation_type=args.representation,
        representations_dir=args.representations_dir,
        top_k=args.top_k,
        rerank_with_cross_encoder=not args.skip_cross_encoder,
        min_retrieval_score=args.min_score,
        output_path=args.output_path,
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
