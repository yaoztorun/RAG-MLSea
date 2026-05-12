# Retrieval Stage — Thesis Interpretation

## Method Overview

| Method | Group | NDCG | delta_Hit@1 | delta_NDCG | Interpretation |
|---|---|---|---|---|---|
| pure_semantic_dense | Pure Semantic | 0.7337 | +0.0 | +0.0 | precision baseline |
| hybrid_type_filtering | Hybrid | 0.7337 | +0.0 | +0.0 | type-control |
| hybrid_type_onehop_filtering | Hybrid | 0.7351 | +0.0 | +0.0013 | graph-neighbourhood hybrid |
| hybrid_predicate_aware_filtering | Hybrid | 0.7375 | +0.0151 | +0.0038 | predicate-aware hybrid |
| optional_rrf_fusion | Optional Fusion | 0.7354 | -0.0226 | +0.0017 | optional recall fusion |
| optional_rrf_symbolic | Optional Fusion | 0.7434 | -0.0075 | +0.0096 | optional fusion + symbolic |

## Framing

The retrieval stage compares three strategy families:

**1. Pure Semantic (Dense)**

- `pure_semantic_dense` is the reference baseline.
- Uses the best pre-retrieval representation per entity type:
  `enriched_metadata` for papers, `dataset_title_only` for datasets,
  `model_predicate_filtered` for models.
- Strong starting point because pre-retrieval tuned the representations.

**2. Semantic + Symbolic Hybrid**

- `hybrid_type_filtering`: applies expected entity-type constraint after dense retrieval.
  Because pre-retrieval outputs are already entity-type-specific, this acts as a
  control. If metrics match the dense baseline, it confirms the collection is pure
  and type filtering adds no information.
- `hybrid_type_onehop_filtering`: applies type filter, then boosts candidates with
  richer graph connections (non-empty tasks/datasets/methods/implementations or
  Linked Entities in source text). Tests whether better graph-connected entities
  are more relevant regardless of question type.
- `hybrid_predicate_aware_filtering`: boosts candidates based on question-type-specific
  predicate signals. Tests whether question intent can guide reranking via symbolic
  predicates (e.g., boost candidates with `tasks` for task-related questions).

**3. Optional RRF Fusion (Exploratory)**

- `optional_rrf_fusion`: fuses multiple representation rankings via Reciprocal Rank
  Fusion (k=60). Trades Hit@1 for broader recall at Hit@10. Not the main strategy.
- `optional_rrf_symbolic`: RRF followed by type filter and predicate-aware boosting.
  Tests whether symbolic post-processing recovers precision lost by fusion.

## What to Look For

- **hybrid_type_filtering == pure_semantic_dense**: expected, because pre-retrieval
  outputs are entity-type-pure. Use as a sanity check, not a performance claim.
- **Hit@1 vs Hit@10 gap**: large gap means the answer exists in the candidate pool
  but not at rank 1. Post-retrieval re-ranking can close this gap.
- **Hard question NDCG**: key diagnostic. If hybrid methods do not improve hard
  questions, the bottleneck is representation quality, not ranking strategy.
- **Dataset entity type**: consistently lower NDCG than papers/models.
  Structural: dataset metadata is sparse in MLSea.

## Important Cautions

- Do not overclaim hybrid improvements. The dense baseline is already strong because
  pre-retrieval selected the best representations.
- Predicate-aware boosting can hurt some question types by reordering away from the
  gold answer when predicate evidence is sparse or ambiguous.
- RRF is included as exploratory fusion to test recall broadening; it is not a main
  thesis claim and should be presented accordingly.
- Re-ranking is NOT part of this stage. Post-retrieval re-ranking operates on the
  fixed top-K candidates produced here.
