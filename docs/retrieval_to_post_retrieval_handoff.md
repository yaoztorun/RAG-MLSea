# Retrieval → Post-Retrieval Handoff

## Why the adapter is needed

The retrieval stage and the post-retrieval stage were developed independently and use different JSON field names for the same concepts. Running one stage's output directly through the other would either crash or silently drop data. The adapter performs a pure schema translation with no algorithmic changes.

## Schema mismatch fixed

| Concept | Retrieval field | Post-retrieval field |
|---|---|---|
| Candidate list key | `candidates` | `results` |
| Entity identifier | `normalized_entity_id` / `entity_id` | `paper_id` |
| Retrieval score | `method_score` / `original_score` | `score` |
| Paper title | `title_or_label` | `title` |
| Rank position | `new_rank` / `original_rank` | `rank` |
| Text snippet | `metadata.source_text` | `source_text` |
| Representation used | `representation_source` | `representation_type` |

The adapter also adds `gold_paper_id = target_entity_iri` at the question level, which post-retrieval evaluation uses for gold-standard comparison.

The fields `entity_type`, `original_rank`, and `original_score` are preserved inside each result for traceability.

## One file per retrieval method

The adapter is run once per retrieval method. Each invocation reads:

```
data/results/retrieval/{method_name}/results.json
```

and writes:

```
data/results/post_retrieval/inputs/{method_name}_post_input.json
```

These files in `data/results/post_retrieval/inputs/` are the sole inputs for the post-retrieval pipeline.

## Top-10 candidates are passed through unchanged

The retrieval stage already limits each question to its top-10 candidates. The adapter preserves that list in its original order without re-ranking, filtering, or expanding it.

## Post-retrieval evaluation is not performed here

This adapter only transforms the data format. It does not invoke the cross-encoder, the LLM generation step, or any evaluation metric. Those steps remain in `src/post_retrieval/pipeline/` and `src/post_retrieval/evaluation/` and must be run separately after the adapter produces its output files.

## Commands

Run the adapter for each retrieval method:

```bash
python -m src.post_retrieval.adapters.retrieval_adapter --method pure_semantic_dense
python -m src.post_retrieval.adapters.retrieval_adapter --method hybrid_type_filtering
python -m src.post_retrieval.adapters.retrieval_adapter --method hybrid_type_onehop_filtering
python -m src.post_retrieval.adapters.retrieval_adapter --method hybrid_predicate_aware_filtering
python -m src.post_retrieval.adapters.retrieval_adapter --method optional_rrf_fusion
python -m src.post_retrieval.adapters.retrieval_adapter --method optional_rrf_symbolic
```

Or with explicit paths:

```bash
python -m src.post_retrieval.adapters.retrieval_adapter \
  --input  data/results/retrieval/pure_semantic_dense/results.json \
  --output data/results/post_retrieval/inputs/pure_semantic_dense_post_input.json
```
