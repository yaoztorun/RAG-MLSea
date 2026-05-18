# RAG-MLSea: Retrieval-Augmented Generation for Machine Learning Question Answering

A master thesis repository implementing a three-stage Retrieval-Augmented Generation (RAG) pipeline
over the MLSea knowledge graph. The system builds entity-centric text representations for papers,
datasets, and models from RDF data, evaluates retrieval strategies across a 500-question machine
learning QA benchmark, and produces ranked candidate lists for downstream post-retrieval processing.

---

## Pipeline Overview

### Pre-Retrieval

Constructs and evaluates text representations for three entity types — papers, datasets, and models —
extracted from the MLSea RDF dump. Fourteen representation strategies are built, embedded using
`sentence-transformers/all-MiniLM-L6-v2`, stored in ChromaDB, and evaluated using ranked retrieval
metrics. The best-performing representations per entity type are selected to serve as the dense
retrieval basis.

Selected representations:

| Entity type | Representations used                          |
|-------------|-----------------------------------------------|
| Papers      | `predicate_filtered`, `enriched_metadata`     |
| Datasets    | `dataset_metadata`                            |
| Models      | `model_metadata`, `model_predicate_filtered`  |

### Retrieval

Six retrieval strategies are evaluated using the selected pre-retrieval representations:

| Method                             | Description                                              |
|------------------------------------|----------------------------------------------------------|
| `pure_semantic_dense`              | Dense retrieval using the best representation (baseline) |
| `hybrid_type_filtering`            | Entity-type-aware score filtering                        |
| `hybrid_type_onehop_filtering`     | Type filtering with graph-neighbourhood boost            |
| `hybrid_predicate_aware_filtering` | Predicate-aware score adjustment                         |
| `optional_rrf_fusion`              | Reciprocal Rank Fusion over selected representations     |
| `optional_rrf_symbolic`            | RRF followed by symbolic type and predicate filtering    |

Each method produces `results.json` and `metrics.json` under `data/results/retrieval/{method}/`.
Aggregate summaries across methods are generated automatically.

### Post-Retrieval

Retrieval outputs are adapted into the post-retrieval input schema via
`src/post_retrieval/adapters/retrieval_adapter.py`. The adapter remaps field names and renames
the candidate list from `candidates` to `results` without modifying either the retrieval or
post-retrieval pipeline. The post-retrieval stage operates only on the fixed top-10 retrieval
candidates and does not query the corpus again.

---

## Repository Structure

```text
config/                          # pipeline configuration (pre_retrieval_config.json)
data/
  intermediate/                  # ChromaDB store, raw extracted records, representations
  questions/                     # 500-question ML QA benchmark dataset
  raw/                           # source RDF dump (pwc_1.nt)
  results/
    pre_retrieval/               # per-representation evaluation results (results.json, top10.json)
    retrieval/                   # per-method retrieval results and aggregate summaries
    post_retrieval/              # post-retrieval inputs and reranking outputs
scripts/                         # standalone artifact and dataset generation scripts
src/
  pre_retrieval/                 # representation construction and evaluation (papers, datasets, models)
  retrieval/                     # six retrieval methods, config, evaluation pipeline
  post_retrieval/                # schema adapter, reranking, context construction
```

---

## Evaluation Metrics

All retrieval stages are evaluated using:

- Hit@1, Hit@5, Hit@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

Metrics are segmented by entity type, question difficulty (easy / medium / hard), and question
category. Only questions whose target entity type matches the retrieval scope contribute to
per-type metrics.

---

## Thesis Artifacts

Pre-retrieval and retrieval results are exported as:

- PDF figures (overall, by difficulty, by category)
- LaTeX tables (booktabs format)
- CSV and JSON summaries

Artifact generation scripts are in `scripts/`.

---

## Running the Pipeline

### Requirements

```bash
pip install -r requirements.txt
```

ChromaDB must be running before any pre-retrieval embedding or evaluation step:

```bash
chroma run --path data/intermediate/chroma
```

### Pre-Retrieval Evaluation

Run all representations for each entity type:

```bash
python -m src.pre_retrieval.papers.scripts.run_all_experiments
python -m src.pre_retrieval.datasets.scripts.run_evaluate_datasets --representation all
python -m src.pre_retrieval.models.scripts.run_evaluate_models --representation all
```

Generate thesis artifacts:

```bash
python scripts/build_pre_retrieval_result_artifacts.py
```

### Retrieval Evaluation

```bash
python -m src.retrieval.scripts.run_all_retrieval_experiments
python scripts/build_retrieval_result_artifacts.py
```

### Retrieval → Post-Retrieval Adapter

```bash
python -m src.post_retrieval.adapters.retrieval_adapter --method pure_semantic_dense
```

Replace `pure_semantic_dense` with any method name from the retrieval stage. Output is written to
`data/results/post_retrieval/inputs/{method}_post_input.json`.

---

## Requirements

- Python 3.10+
- `sentence-transformers` (model: `sentence-transformers/all-MiniLM-L6-v2`)
- `chromadb`
- `rdflib`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`
