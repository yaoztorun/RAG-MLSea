# RAG-MLSea

Master thesis repository for local pre-retrieval and retrieval experiments on MLSea / Papers with Code metadata.

## Active local workflow

The active local workflow supports multiple entity types for pre-retrieval and retrieval:

### Paper pipeline (fully implemented)

1. canonical paper extraction from `data/raw/pwc_1.nt`
2. curated subset construction from `papers_master.jsonl`
3. representation building
4. Chroma embedding storage
5. retrieval evaluation
6. top-10 export per question for later post-retrieval use

### Dataset pipeline

1. canonical dataset extraction from `data/raw/pwc_1.nt`
2. representation building
3. Chroma embedding storage
4. retrieval evaluation

Local experiments use the curated subset by default for papers. Full-corpus runs are reserved for later VSC execution.

## Active pipeline layout

```text
src/pre_retrieval/
  config.py
  utils.py
  raw_papers/
    build_paper_records.py
    build_curated_subset.py
    inspect_paper_predicates.py
  raw_datasets/
    build_dataset_records.py
  chunking/
    build_representations.py
    build_dataset_representations.py
    papers/
      build_title_only_chunks.py
      build_abstract_only_chunks.py
      build_title_abstract_chunks.py
      build_enriched_paper_chunks.py
      build_predicate_filtered_chunks.py
      build_one_hop_paper_chunks.py
    datasets/
      build_dataset_title_only.py
      build_dataset_metadata.py
      build_dataset_predicate_filtered.py
  embeddings/
    embed_and_store.py
    embedder.py
    vector_store.py
  retrieval/
    retrieve.py
  evaluation/
    evaluate_retrieval.py
    aggregate_results.py
  scripts/
    run_build_records.py
    run_build_subset.py
    run_build_representations.py
    run_embed_store.py
    run_evaluate.py
    run_all_experiments.py
    run_aggregate_results.py
    run_build_datasets.py
    run_build_dataset_representations.py
    run_embed_store_datasets.py
    run_evaluate_datasets.py
src/retrieval/
  evaluation/
src/post_retrieval/
  evaluation/
archive/
```

## Curated subset rules

The shared local subset is built from:

- canonical paper records: `data/intermediate/raw_papers/papers_master.jsonl`
- questions: `data/questions/ml_questions_dataset.json`

Default local subset config:

```json
"corpus_subset": {
  "enabled": true,
  "max_papers": 200000,
  "include_gold_targets": true
}
```

Subset builder behavior:

- normalize each `target_entity_iri` from the question dataset to a canonical `paper_id`
- distinguish paper (`scientificWork`) targets from non-paper targets
- include all paper gold-target papers found in the canonical records first
- fill the remaining capacity from the canonical records in their existing order
- stop at 200,000 papers total unless the gold-target set is already larger
- reuse the same subset for all representations to keep comparisons fair
- report non-paper gold targets separately instead of treating them as missing papers

Outputs:

- `data/intermediate/raw_papers/papers_subset_200k.jsonl`
- `data/intermediate/raw_papers/subset_stats.json`

## Representation strategies

### Paper representations

All local experiments use the same curated subset for:

1. `title_only`
2. `abstract_only`
3. `title_abstract`
4. `predicate_filtered`
5. `enriched_metadata`
6. `one_hop`

### Dataset representations

1. `dataset_title_only` — uses label/title only
2. `dataset_metadata` — uses title/label + description + keywords + tasks + year
3. `dataset_predicate_filtered` — uses selected useful predicates/fields for datasets

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the RDF dump at `data/raw/pwc_1.nt`, or pass `--input-path` explicitly.

Start Chroma before embedding or evaluation:

```bash
chroma run --path data/intermediate/chroma
```

## Paper pipeline — default local run order

Build canonical paper records:

```bash
python -m src.pre_retrieval.scripts.run_build_records
```

Build the shared curated subset:

```bash
python -m src.pre_retrieval.scripts.run_build_subset
```

Build one representation from the shared subset:

```bash
python -m src.pre_retrieval.scripts.run_build_representations --representation title_only
```

Embed and store one representation:

```bash
python -m src.pre_retrieval.scripts.run_embed_store --representation title_only
```

Evaluate one representation and export top-10 documents:

```bash
python -m src.pre_retrieval.scripts.run_evaluate --representation title_only
```

Run the full local comparison workflow in the active order:

```bash
python -m src.pre_retrieval.scripts.run_all_experiments
```

## Dataset pipeline — run order

Build canonical dataset records:

```bash
python -m src.pre_retrieval.scripts.run_build_datasets
```

Build dataset representations (one or all):

```bash
python -m src.pre_retrieval.scripts.run_build_dataset_representations --representation all
```

Embed and store dataset representations:

```bash
python -m src.pre_retrieval.scripts.run_embed_store_datasets --representation dataset_title_only
python -m src.pre_retrieval.scripts.run_embed_store_datasets --representation dataset_metadata
python -m src.pre_retrieval.scripts.run_embed_store_datasets --representation dataset_predicate_filtered
```

Evaluate dataset representations:

```bash
python -m src.pre_retrieval.scripts.run_evaluate_datasets --representation dataset_title_only
python -m src.pre_retrieval.scripts.run_evaluate_datasets --representation dataset_metadata
python -m src.pre_retrieval.scripts.run_evaluate_datasets --representation dataset_predicate_filtered
```

## Shared aggregation

Regenerate the shared comparison summaries from whatever per-representation results already exist across all entity types:

```bash
python -m src.pre_retrieval.scripts.run_aggregate_results
```

This aggregation reads from:
- `data/retrieval_results/paper_results/*/results.json`
- `data/retrieval_results/dataset_results/*/results.json`
- future `*_results/` entity type folders

Useful overrides:

- `--max-papers N`
- `--disable-subset`
- `--force-rebuild`
- `--limit N`
- `--skip-existing`

When subset mode is enabled, `run_all_experiments` rebuilds the shared subset first and then runs all six paper representations against that same subset.

## Retrieval outputs

### Structure

Results are now organized by entity type:

```text
data/retrieval_results/
  paper_results/
    title_only/results.json, top10.json
    abstract_only/results.json, top10.json
    title_abstract/results.json, top10.json
    predicate_filtered/results.json, top10.json
    enriched_metadata/results.json, top10.json
    one_hop/results.json, top10.json
  dataset_results/
    dataset_title_only/results.json, top10.json
    dataset_metadata/results.json, top10.json
    dataset_predicate_filtered/results.json, top10.json
  summary.json
  summary.md
  summary.csv
  summary_by_difficulty.json
  summary_by_category.json
```

### Shared summaries

Shared summaries at the root of `data/retrieval_results/` combine results from all entity types:

- `summary.json` — rows include `entity_type` and `representation` fields
- `summary.md` — markdown table with entity type column
- `summary.csv` — CSV with entity type column
- `summary_by_difficulty.json` — segmented by difficulty across all entity types
- `summary_by_category.json` — segmented by category across all entity types

This structure scales to future entity types (models, implementations, algorithms) by adding `*_results/` folders.

### Per-representation results

`results.json` records:

- `entity_type`
- `diagnostics`
- `metrics`
- `metrics_by_difficulty`
- `metrics_by_category`
- `per_question`

The segmented outputs keep the entity-centered evaluation rule for retrieval metrics: only answerable questions whose target matches the entity type contribute to `Hit@k`, `MRR`, and `NDCG`.

`top10.json` stores the top-10 retrieved documents per question together with retrieval scores, source text, and canonical metadata fields when available.

`run_aggregate_results` reads existing `data/retrieval_results/{entity_type}_results/{representation}/results.json` files, preserves the configured representation order, skips missing results gracefully, and does not rerun embedding or evaluation.

If you want optional abstention analysis for unanswerable questions, set `evaluation.abstention_score_threshold` in `config/pre_retrieval_config.json` or pass `--abstention-score-threshold` to `run_evaluate`.

## Archive policy

Old GraphDB / SPARQL-dependent assets, duplicate scripts, and stale demo outputs are moved under `archive/` instead of being deleted.

## Notes

- `data/raw/pwc_1.nt` is the source of truth for canonical extraction
- the active pipeline is offline-first and does not depend on live GraphDB / SPARQL during pre-retrieval or retrieval
- retrieval metrics remain `Hit@1`, `Hit@5`, `Hit@10`, `MRR`, and `NDCG`
- papers are fully implemented
- datasets are the next entity type
- shared summaries now support multiple entity types
- post-retrieval remains in the repository, but it is not the center of the active local workflow
