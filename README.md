# MLSea KG-RAG Retrieval Pipeline

A master thesis project implementing a Retrieval-Augmented Generation pipeline over the [MLSea](https://mlsea.eu) knowledge graph. The system builds entity-centric text representations for paper, dataset, and model entities from local RDF data, embeds them into ChromaDB, and evaluates six retrieval strategies on a 500-question machine-learning QA benchmark. Retrieved candidates are prepared as structured inputs for downstream post-retrieval processing.

---

## Pipeline

1. Parse local MLSea/Papers-with-Code RDF data and build canonical entity records
2. Construct entity-centric textual representations (multiple strategies per entity type)
3. Generate embeddings and populate ChromaDB vector indexes
4. Run retrieval experiments (dense, hybrid, and RRF-based methods)
5. Evaluate and export retrieval results, figures, and LaTeX tables
6. Adapt retrieval outputs into post-retrieval input format

---

## Repository Structure

```text
config/                      # pipeline configuration (pre_retrieval_config.json)
data/
  raw/                       # source RDF dump — not tracked in Git
  intermediate/              # ChromaDB indexes, extracted records, representations
  questions/                 # 500-question ML QA benchmark
  results/
    pre_retrieval/           # per-representation evaluation results
    retrieval/               # per-method retrieval results and aggregate summaries
    post_retrieval/          # post-retrieval input files and outputs
scripts/                     # standalone artifact and figure generation scripts
src/
  pre_retrieval/             # representation construction and evaluation
  retrieval/                 # six retrieval methods, config, evaluation pipeline
  post_retrieval/            # schema adapter, reranking, context construction
figs/results_final/          # final PDF/PNG figures and LaTeX snippets
docs/chapter4_artifacts/     # thesis tables and supporting artifacts
```

> Large raw RDF data, ChromaDB indexes, and generated result files are not tracked in Git.

---

## Requirements

- Python 3.10+
- Install dependencies:

```
pip install -r requirements.txt
```

Key packages: `chromadb`, `sentence-transformers`, `rdflib`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`.

The embedding model used throughout is `sentence-transformers/all-MiniLM-L6-v2`.

---

## Data Setup

The raw MLSea/Papers-with-Code RDF dump is expected at:

```
data/raw/pwc_1.nt
```

This file is not included in the repository. The questions benchmark is at:

```
data/questions/ml_questions_dataset.json
```

---

## Running the Pipeline

### Start ChromaDB

ChromaDB runs as a local HTTP server on port 8000. Start it before any embedding or evaluation step and keep it running in a separate terminal:

```
chroma run --path data/intermediate/chroma
```

---

### 1. Build pre-retrieval representations and embeddings

Run the full pre-retrieval pipeline for each entity type. This step is expensive on first run — it builds textual representations, generates embeddings, and stores them in ChromaDB.

```
py -m src.pre_retrieval.papers.scripts.run_all_experiments
py -m src.pre_retrieval.datasets.scripts.run_evaluate_datasets --representation all
py -m src.pre_retrieval.models.scripts.run_evaluate_models --representation all
```

Generate pre-retrieval thesis artifacts (figures, tables, CSVs):

```
py scripts/build_pre_retrieval_result_artifacts.py
```

---

### 2. Run retrieval experiments

Runs all six retrieval methods and writes results under `data/results/retrieval/`:

```
py -m src.retrieval.scripts.run_all_retrieval_experiments --force
```

Key outputs per method: `results.json`, `metrics.json`. Aggregate outputs: `summary.csv`, `summary_by_entity_type.json`, `summary_by_difficulty.json`, `summary_by_question_type.json`.

---

### 3. Generate retrieval figures

```
py scripts/generate_results_final_figures.py --force
```

Outputs written to `figs/results_final/` (PDF + PNG figures, `LATEX_SNIPPETS.tex`, `FIGURE_AUDIT.md`).

---

### 4. Generate post-retrieval inputs

Convert a single method's retrieval output into the post-retrieval input schema:

```
py -m src.post_retrieval.adapters.retrieval_adapter --method pure_semantic_dense
```

To convert all six methods (PowerShell):

```powershell
$methods = @(
  "pure_semantic_dense",
  "hybrid_type_filtering",
  "hybrid_type_onehop_filtering",
  "hybrid_predicate_aware_filtering",
  "optional_rrf_fusion",
  "optional_rrf_symbolic"
)
foreach ($m in $methods) {
  Write-Host "`n=== $m ==="
  py -m src.post_retrieval.adapters.retrieval_adapter --method $m
}
```

Output files: `data/results/post_retrieval/inputs/{method}_post_input.json`.

---

### 5. Post-retrieval evaluation

Post-retrieval answer generation and evaluation scripts are located under `src/post_retrieval/` (`scripts/`, `evaluation/`, `generation/`, `pipeline/` subdirectories).

---

## Retrieval Methods

| Method | Description |
|---|---|
| `pure_semantic_dense` | Dense semantic retrieval over the best representation per entity type (baseline) |
| `hybrid_type_filtering` | Structural control — entity-type collections already enforce type specificity, so this is equivalent to the dense baseline |
| `hybrid_type_onehop_filtering` | Extends the dense baseline with a one-hop graph-neighbourhood richness boost |
| `hybrid_predicate_aware_filtering` | Soft reranking using lexical overlap between the question and entity predicate fields (ε = 0.02) |
| `optional_rrf_fusion` | Reciprocal Rank Fusion (k = 60) over multiple representation indexes for improved top-10 recall |
| `optional_rrf_symbolic` | RRF fusion followed by symbolic predicate-aware reranking; strongest overall method by NDCG |

---

## Outputs

| Output | Location |
|---|---|
| ChromaDB vector indexes | `data/intermediate/chroma/` |
| Pre-retrieval evaluation results | `data/results/pre_retrieval/` |
| Retrieval results and summaries | `data/results/retrieval/` |
| Post-retrieval input files | `data/results/post_retrieval/inputs/` |
| Final figures (PDF/PNG) | `figs/results_final/` |
| LaTeX figure snippets | `figs/results_final/LATEX_SNIPPETS.tex` |
| Thesis tables (LaTeX) | `docs/chapter4_artifacts/tables/` |

---

## Reproducibility Notes

- All results depend on the local RDF dump, the question dataset, the representation scripts, the embedding model, and the ChromaDB collections.
- If only retrieval reranking logic changes, embeddings do not need to be regenerated — rerun retrieval experiments and the post-retrieval adapter.
- If representations or embeddings change, a full rebuild (embed → retrieval → adapter) is required.

---

## Thesis Context

This repository supports a master thesis on knowledge-graph-based retrieval for machine-learning question answering over the MLSea RDF dataset.
