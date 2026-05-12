# Retrieval Stage

This module implements the retrieval stage of the RAG-MLSea pipeline.

## Position in the Pipeline

```
Pre-retrieval  →  Retrieval (this module)  →  Post-retrieval
```

- **Pre-retrieval**: builds and evaluates entity text representations; saves top-K results per representation.
- **Retrieval**: consumes those saved top-K outputs and compares candidate-generation strategies.
- **Post-retrieval**: takes the fixed top-K candidates from this stage and applies re-ranking, context construction, and generation.

> **Re-ranking does not belong here.** Retrieval produces fixed top-K candidates. Re-ranking is post-retrieval.

## Input

Pre-retrieval top-10 outputs (read-only):

```
data/results/pre_retrieval_results/
  paper_results/{representation}/top10.json
  dataset_results/{representation}/top10.json
  model_results/{representation}/top10.json
```

Questions:

```
data/questions/ml_questions_dataset.json
```

## Output

```
data/results/retrieval/
  {method_name}/
    results.json     — per-question results with candidates and metrics
    metrics.json     — aggregated metrics (overall + by difficulty/entity_type/question_type)
  summary.json
  summary.csv
  summary.md
  summary_by_difficulty.json
  summary_by_entity_type.json
  summary_by_question_type.json
  summary_delta_vs_dense.json   — per-metric delta of each method vs pure_semantic_dense
  summary_delta_vs_dense.csv
  thesis_tables/
    retrieval_main_comparison.csv
    retrieval_by_difficulty_ndcg.csv
    retrieval_precision_recall_tradeoff.csv
    retrieval_by_entity_type_ndcg.csv
    retrieval_by_question_type_ndcg.csv
    retrieval_interpretation.md
```

## Method Families

### 1. Pure Semantic (Dense)

**`pure_semantic_dense`**

Loads the best representation per entity type and returns the pre-retrieval top-10 as candidates:
- Paper → `enriched_metadata`
- Dataset → `dataset_title_only`
- Model → `model_predicate_filtered`

This is the reference baseline. It is strong because pre-retrieval already selected the best representation.

---

### 2. Semantic + Symbolic Hybrid

**`hybrid_type_filtering`**

Starts from `pure_semantic_dense` candidates, then filters to only candidates whose entity type matches the expected type inferred from `target_entity_iri`. Falls back to unfiltered if no candidates match.

> Because pre-retrieval outputs are already entity-type-specific (paper questions only return papers, etc.), this method typically produces identical results to `pure_semantic_dense`. It serves as a **control** confirming the collection purity, not a performance gain. If metrics are identical to the dense baseline, this is expected and correct.

**`hybrid_type_onehop_filtering`**

Applies type filter first, then boosts candidates with richer graph connections. The richness score counts non-empty metadata fields: `tasks`, `datasets`, `methods`, `metrics`, `implementations`, plus a bonus for "Linked Entities" in source text. Boost is question-type-agnostic.

Hypothesis: better-connected entities in the ML knowledge graph are more likely to be the answer, independent of the specific question predicate.

**`hybrid_predicate_aware_filtering`**

Starts from `pure_semantic_dense` candidates, then applies lightweight question-type-specific boosting. Examples:
- Task-related question types → boost candidates with non-empty `tasks`
- `paper_to_implementation` → boost candidates with non-empty `implementations`
- `repository_to_model` / `model_family_variant` → boost candidates with "Linked Entities" in source text

Falls back to original order if no predicate evidence is found, with a warning.

---

### 3. Optional Exploratory Fusion

These methods are included as exploratory additions. They are not the main thesis claim. RRF broadens recall at the cost of Hit@1.

**`optional_rrf_fusion`**

Applies Reciprocal Rank Fusion over multiple representations:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

Default `k = 60`. Fusion groups:
- Paper: `enriched_metadata`, `predicate_filtered`, `one_hop`
- Dataset: `dataset_title_only`, `dataset_enriched_metadata`
- Model: `model_predicate_filtered`, `model_enriched_metadata`

**`optional_rrf_symbolic`**

Applies `optional_rrf_fusion` first, then `hybrid_type_filtering`, then `hybrid_predicate_aware_filtering`. Tests whether symbolic post-processing recovers precision lost by fusion.

---

## Backward-Compatible Method Aliases

Old method names are still accepted by scripts and `run_method()`:

| Old name | New name |
|---|---|
| `dense_baseline` | `pure_semantic_dense` |
| `type_filtering` | `hybrid_type_filtering` |
| `predicate_aware_filtering` | `hybrid_predicate_aware_filtering` |
| `rrf_fusion` | `optional_rrf_fusion` |
| `rrf_symbolic` | `optional_rrf_symbolic` |

---

## Evaluation Metrics

- **Hit@1**, **Hit@5**, **Hit@10**: fraction of questions where gold appears in top K.
- **MRR**: mean reciprocal rank.
- **NDCG**: normalised discounted cumulative gain at top-K (primary metric).

Gold is `target_entity_iri` (normalised / URL-decoded). Only answerable questions with a valid IRI are included in metric computation (15 unanswerable questions are preserved in `results.json` but excluded from averaging).

---

## Scripts

```bash
# Run all methods and aggregate
python -m src.retrieval.scripts.run_all_retrieval_experiments

# Run a single method (new or legacy name)
python -m src.retrieval.scripts.run_retrieval_stage --method pure_semantic_dense
python -m src.retrieval.scripts.run_retrieval_stage --method dense_baseline  # alias

# Force rerun
python -m src.retrieval.scripts.run_retrieval_stage --method pure_semantic_dense --force

# Re-evaluate from saved results
python -m src.retrieval.scripts.run_evaluate_retrieval_stage --method all
```

All scripts accept `--help` for full option listings.
