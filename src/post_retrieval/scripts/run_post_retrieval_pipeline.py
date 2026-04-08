from __future__ import annotations

import argparse

from src.post_retrieval.pipeline import (
    build_context_payload,
    build_paper_id_lookup,
    build_representation_lookup,
    load_canonical_paper_records,
    load_cross_encoder,
    load_representation_records,
    load_retrieval_payload,
    resolve_question_retrieval_entry,
    resolve_retrieval_results_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline post-retrieval context for one question.")
    parser.add_argument("--representation", default="title_only")
    parser.add_argument("--retrieval-results-path")
    parser.add_argument("--question-id")
    parser.add_argument("--question-index", type=int)
    parser.add_argument("--question-text")
    parser.add_argument("--papers-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--representations-dir", default="data/intermediate/representations")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    args = parser.parse_args()

    retrieval_results_path = args.retrieval_results_path or str(resolve_retrieval_results_path(args.representation))
    retrieval_payload = load_retrieval_payload(retrieval_results_path)
    question_entry = resolve_question_retrieval_entry(
        retrieval_payload,
        question_id=args.question_id,
        question_index=args.question_index,
        question_text=args.question_text,
        default_to_first=True,
    )
    if question_entry is None:
        raise ValueError("Could not resolve a question entry from the retrieval results payload.")

    question = question_entry["question"]
    question_id = question_entry.get("question_id", "")
    target_entity_iri = question_entry.get("target_entity_iri", "")
    results = question_entry.get("results", [])
    print(f"# question_id: {question_id}")
    print(f"# target_entity_iri: {target_entity_iri}")
    paper_lookup = build_paper_id_lookup(load_canonical_paper_records(args.papers_path))
    representation_lookup = build_representation_lookup(
        load_representation_records(args.representation, representations_dir=args.representations_dir)
    )
    cross_encoder = None if args.skip_cross_encoder else load_cross_encoder()
    payload = build_context_payload(
        question,
        results,
        paper_lookup,
        representation_lookup=representation_lookup,
        cross_encoder=cross_encoder,
        use_cross_encoder=not args.skip_cross_encoder,
        min_retrieval_score=args.min_score,
        top_k=args.top_k,
    )
    print(payload["context"])


if __name__ == "__main__":
    main()
