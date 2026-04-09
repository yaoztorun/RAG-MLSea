# RAG-MLSea

Master thesis repository for local pre-retrieval and retrieval experiments on MLSea / Papers with Code metadata.

## Active local workflow

The active local workflow is now focused on the pre-retrieval and retrieval stages only:

1. canonical paper extraction from `data/raw/pwc_1.nt`
2. curated subset construction from `papers_master.jsonl`
3. representation building
4. Chroma embedding storage
5. retrieval evaluation
6. top-10 export per question for later post-retrieval use

Local experiments use the curated subset by default. Full-corpus runs are reserved for later VSC execution.

## Active pipeline layout

```text
src/pre_retrieval/
  config.py
  raw_papers/
    build_paper_records.py
    build_curated_subset.py
    inspect_paper_predicates.py
  chunking/
    build_representations.py
    papers/
      build_title_only_chunks.py
      build_abstract_only_chunks.py
      build_title_abstract_chunks.py
      build_enriched_paper_chunks.py
      build_predicate_filtered_chunks.py
      build_one_hop_paper_chunks.py
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

All local experiments use the same curated subset for:

1. `title_only`
2. `abstract_only`
3. `title_abstract`
4. `predicate_filtered`
5. `enriched_metadata`
6. `one_hop`

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

## Default local run order

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

Regenerate the shared comparison summaries from whatever per-representation results already exist:

```bash
python -m src.pre_retrieval.scripts.run_aggregate_results
```

Run the full local comparison workflow in the active order:

```bash
python -m src.pre_retrieval.scripts.run_all_experiments
```

Useful overrides:

- `--max-papers N`
- `--disable-subset`
- `--force-rebuild`
- `--limit N`
- `--skip-existing`

When subset mode is enabled, `run_all_experiments` rebuilds the shared subset first and then runs all six representations against that same subset.

## Retrieval outputs

Each representation now writes into its own result folder:

- `data/retrieval_results/title_only/`
- `data/retrieval_results/abstract_only/`
- `data/retrieval_results/title_abstract/`
- `data/retrieval_results/enriched_metadata/`
- `data/retrieval_results/predicate_filtered/`
- `data/retrieval_results/one_hop/`

Each representation folder contains:

- `results.json`
- `top10.json`
- any future representation-specific summaries

`results.json` now keeps the existing overall metrics and additionally records:

- `diagnostics`
- `metrics_by_difficulty`
- `metrics_by_category`
- `per_question`

The segmented outputs keep the paper-centered evaluation rule for retrieval metrics: only answerable paper-target questions contribute to `Hit@k`, `MRR`, and `NDCG`, while non-paper targets and unanswerable questions are reported explicitly in diagnostics and segmented counts.

Aggregate summaries remain at:

- `data/retrieval_results/summary.json`
- `data/retrieval_results/summary.md`
- `data/retrieval_results/summary.csv`
- `data/retrieval_results/summary_by_difficulty.json`
- `data/retrieval_results/summary_by_category.json`

`top10.json` stores the top-10 retrieved documents per question together with retrieval scores, source text, and canonical metadata fields such as authors, publication year, tasks, datasets, methods, metrics, implementations, and keywords when available.

`run_aggregate_results` reads only existing `data/retrieval_results/{representation}/results.json` files, preserves the configured representation order, skips missing results gracefully, and does not rerun embedding or evaluation.

Current retrieval evaluation is paper-centered: it evaluates only answerable questions whose gold target is a `scientificWork` / paper and explicitly skips answerable non-paper targets.

If you want optional abstention analysis for unanswerable questions, set `evaluation.abstention_score_threshold` in `config/pre_retrieval_config.json` or pass `--abstention-score-threshold` to `run_evaluate`.

## Archive policy

Old GraphDB / SPARQL-dependent assets, duplicate scripts, and stale demo outputs are moved under `archive/` instead of being deleted.

## Notes

- `data/raw/pwc_1.nt` is the source of truth for canonical extraction
- the active pipeline is offline-first and does not depend on live GraphDB / SPARQL during pre-retrieval or retrieval
- retrieval metrics remain `Hit@1`, `Hit@5`, `Hit@10`, `MRR`, and `NDCG`
- post-retrieval remains in the repository, but it is not the center of the active local workflow
