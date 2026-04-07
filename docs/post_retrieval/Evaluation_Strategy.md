# Offline Post-Retrieval Evaluation Strategy

The evaluation stage is now aligned with the offline architecture.

## Retrieval evaluation

`src/post_retrieval/evaluation/evaluate_retrieval.py` measures how much the post-retrieval stage improves the existing retrieval outputs.

It reads:

- `data/retrieval_results/{representation}_results.json`
- `data/questions/ml_questions_dataset.json`
- `data/intermediate/raw_papers/papers_master.jsonl`
- optionally `data/intermediate/representations/{representation}.jsonl`

The evaluation compares:

1. the original retrieval top-k from the current pipeline
2. the top-k after offline post-retrieval filtering and optional cross-encoder reranking

This preserves the useful ablation idea from `esat_branch` without re-querying GraphDB.

## Generation evaluation

`src/post_retrieval/evaluation/evaluate_generation.py` evaluates answers generated from offline context.

It uses:

- `text_answer` from `data/questions/ml_questions_dataset.json` as ground truth
- Semantic Answer Similarity (SAS) with `sentence-transformers`
- optional ROUGE-L if `rouge-score` is installed in the runtime environment

The generator is injected as a callable so the evaluation logic stays independent from a specific LLM backend.

## Output location

Recommended generated outputs belong under `data/intermediate/post_retrieval/` or another ignored generated-data path, not in version-controlled root artifacts.
