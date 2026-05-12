# Retrieval Stage Plan

## Goal
Compare candidate generation strategies after pre-retrieval representation evaluation.

Pre-retrieval answers: "What should be embedded?"
Retrieval answers: "How should top-K candidates be generated?"

**Re-ranking is NOT part of this stage.** Re-ranking operates on the fixed top-K
candidates produced here and belongs to post-retrieval.

## Input
Pre-retrieval top-K outputs from:

`data/results/pre_retrieval_results/`

(fallback paths: `data/results/pre_retrieval/`, `data/retrieval_results/`)

## Output

`data/results/retrieval/`

Each method produces:
- `results.json` — per-question results with candidates and metrics
- `metrics.json` — aggregated metrics (overall + by difficulty/entity_type/question_type)

Global summaries:
- `summary.json`, `summary.csv`, `summary.md`
- `summary_by_difficulty.json`, `summary_by_entity_type.json`, `summary_by_question_type.json`
- `summary_delta_vs_dense.json`, `summary_delta_vs_dense.csv`
- `thesis_tables/` — visualization-ready CSVs and interpretation markdown

## Method Families

### Family 1 — Pure Semantic (Dense)

**`pure_semantic_dense`** (formerly `dense_baseline`)

Uses the best pre-retrieval representation per entity type:
- Paper: `enriched_metadata`
- Dataset: `dataset_title_only`
- Model: `model_predicate_filtered`

This is the baseline. It is strong because pre-retrieval already selected the best representations.

### Family 2 — Semantic + Symbolic Hybrid

**`hybrid_type_filtering`** (formerly `type_filtering`)

Filters candidates by expected entity type. Because pre-retrieval outputs are already
entity-type-specific, this serves as a **control/no-op** that confirms collection purity.
If metrics match `pure_semantic_dense`, that is expected and correct.

**`hybrid_type_onehop_filtering`** *(new)*

Type filter + general graph-connectivity boosting. Boosts candidates with richer one-hop
connections (non-empty tasks/datasets/methods/implementations or Linked Entities in source
text). Question-type-agnostic boost. Tests whether better-connected graph entities are
more likely to be the answer.

**`hybrid_predicate_aware_filtering`** (formerly `predicate_aware_filtering`)

Question-type-specific predicate boosting without a type filter. Tests whether question
intent can guide reranking via symbolic predicates.

### Family 3 — Optional Exploratory Fusion

These are included as exploratory additions and are not the main thesis claim. RRF
broadens recall (Hit@10) at the cost of precision (Hit@1).

**`optional_rrf_fusion`** (formerly `rrf_fusion`)

Reciprocal Rank Fusion over multiple representations: `score(d) = Σ 1/(k + rank_i(d))`, k=60.

Fusion groups:
- Papers: `enriched_metadata`, `predicate_filtered`, `one_hop`
- Datasets: `dataset_title_only`, `dataset_enriched_metadata`
- Models: `model_predicate_filtered`, `model_enriched_metadata`

**`optional_rrf_symbolic`** (formerly `rrf_symbolic`)

RRF first, then type filtering, then predicate-aware boosting.

## Evaluation
Metrics: Hit@1, Hit@5, Hit@10, MRR, NDCG (primary: NDCG)

Segmented by: difficulty, entity type, question type

Primary comparison: delta vs `pure_semantic_dense`.