from __future__ import annotations

import argparse
import json

from src.post_retrieval.evaluation import evaluate_generation
from src.post_retrieval.generation import generate_rag_answer, judge_rag_answer, load_generation_model
from src.post_retrieval.pipeline import resolve_retrieval_results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline answer generation from post-retrieval context.")
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--retrieval-results-path")
    parser.add_argument("--papers-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--questions-path", default="data/questions/ml_questions_dataset.json")
    parser.add_argument("--representations-dir", default="data/intermediate/representations")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-path", default="data/intermediate/post_retrieval/generation_evaluation.json")
    args = parser.parse_args()

    retrieval_results_path = args.retrieval_results_path or str(resolve_retrieval_results_path(args.representation))
    model, tokenizer, device = load_generation_model(model_id=args.model_id, device=args.device)

    def generator(question: str, context: str) -> str:
        return generate_rag_answer(question, context, model=model, tokenizer=tokenizer, device=device)

    def judge(ground_truth: str, generated_answer: str) -> int:
        return judge_rag_answer(ground_truth, generated_answer, model=model, tokenizer=tokenizer, device=device)

    payload = evaluate_generation(
        retrieval_results_path=retrieval_results_path,
        generator_fn=generator,
        judge_fn=judge,
        canonical_records_path=args.papers_path,
        questions_path=args.questions_path,
        representation_type=args.representation,
        representations_dir=args.representations_dir,
        top_k=args.top_k,
        min_retrieval_score=args.min_score,
        rerank_with_cross_encoder=not args.skip_cross_encoder,
        limit=args.limit,
        output_path=args.output_path,
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
