# Methodology Chapter — Writing Execution Plan

**Thesis:** Retrieval-Augmented Generation over Machine Learning Knowledge Graphs  
**Institution:** KU Leuven  
**Target Chapter:** Chapter 3 — Methodology  
**Scope of this plan:** Pre-Retrieval and Retrieval phases  
**Date generated:** 2026-05-05  
**All facts below are verified against the repository unless marked TODO.**

---

## Table of Contents

1. [Repository Understanding](#1-repository-understanding)
2. [Methodology Chapter Structure](#2-methodology-chapter-structure)
3. [Pre-Retrieval Phase: Thesis Explanation Plan](#3-pre-retrieval-phase-thesis-explanation-plan)
4. [Retrieval Phase: Thesis Explanation Plan](#4-retrieval-phase-thesis-explanation-plan)
5. [Figures and Tables Plan](#5-figures-and-tables-plan)
6. [Thesis-Ready Writing Blocks](#6-thesis-ready-writing-blocks)
7. [Missing Information Checklist](#7-missing-information-checklist)
8. [Final Recommended Writing Order](#8-final-recommended-writing-order)

---

## 1. Repository Understanding

### 1.1 Overview

The repository implements a three-stage Retrieval-Augmented Generation (RAG) pipeline specifically designed for machine-learning-domain question answering over the MLSea RDF knowledge graph. The system converts RDF graph entities into textual representations, embeds them into a vector store, retrieves relevant candidates for natural-language questions, and (as future/scaffold work) generates answers using a language model.

The pipeline is fully offline: it has no dependency on a running GraphDB or SPARQL endpoint at inference time. All KG data is pre-processed, extracted, and stored locally.

### 1.2 Repository Layout

```
RAG-MLSea/
├── config/
│   └── pre_retrieval_config.json          ← all pipeline hyper-parameters
├── data/
│   ├── raw/
│   │   └── pwc_1.nt                       ← source RDF (6.4 GB, 26,606,202 triples)
│   ├── questions/
│   │   └── ml_questions_dataset.json      ← 280 evaluation questions
│   ├── intermediate/
│   │   ├── raw_papers/                    ← extracted paper records (JSONL)
│   │   ├── raw_datasets/                  ← extracted dataset records (JSONL)
│   │   ├── raw_models/                    ← extracted model records (JSONL)
│   │   ├── representations/               ← text representation chunks (JSONL per strategy)
│   │   └── chroma/                        ← vector store (8.2 GB SQLite, 18 collections)
│   └── results/
│       ├── pre_retrieval_results/         ← per-representation retrieval eval results
│       ├── retrieval/                     ← retrieval-stage method comparison results
│       ├── thesis_tables/                 ← CSV files for thesis tables
│       └── thesis_figures/               ← generated PDF/PNG plots
├── src/
│   ├── pre_retrieval/
│   │   ├── shared/                        ← config, embedder, vector_store, retrieve, evaluate, aggregate
│   │   ├── papers/raw/                    ← RDF extraction for papers
│   │   ├── papers/chunking/               ← 6 representation builders for papers
│   │   ├── datasets/raw/                  ← RDF extraction for datasets
│   │   ├── datasets/chunking/             ← 4 representation builders for datasets
│   │   ├── models/raw/                    ← RDF extraction for models
│   │   └── models/chunking/               ← 4 representation builders for models
│   ├── retrieval/
│   │   ├── config.py                      ← best representations, RRF groups, method list
│   │   ├── pipeline.py                    ← run all methods, save results
│   │   ├── dense_baseline.py              ← pure_semantic_dense
│   │   ├── filtering.py                   ← three hybrid methods
│   │   ├── rrf.py                         ← two RRF fusion methods
│   │   ├── result_schema.py               ← metric computation (Hit@k, MRR, NDCG)
│   │   └── evaluate_retrieval_stage.py    ← aggregate + segment metrics
│   └── post_retrieval/
│       ├── pipeline/context_builder.py    ← context assembly + cross-encoder re-ranking
│       ├── generation/llama_generation.py ← TinyLlama answer generation
│       └── evaluation/                    ← SAS, ROUGE-L, LLM judge (scaffold)
├── docs/post_retrieval/                   ← planning and methodology documents
└── archive/                               ← legacy GraphDB-dependent code
```

### 1.3 Data Sources

| Item | Path | Size | Description |
|------|------|------|-------------|
| MLSea KG | `data/raw/pwc_1.nt` | 6.4 GB, 26,606,202 triples | RDF export of Papers with Code in N-Triples format; source of truth for all entity data |
| Question dataset | `data/questions/ml_questions_dataset.json` | ~3.6 MB, 280 questions | Manually curated evaluation questions with gold entity IRIs |
| Paper corpus subset | `data/intermediate/raw_papers/papers_subset_200k.jsonl` | ~680 MB | 200,000 papers extracted from KG; curated to include all gold targets |
| Vector store | `data/intermediate/chroma/chroma.sqlite3` | 8.2 GB | 18 Chroma collections (one per representation); HNSW index, cosine metric |

### 1.4 KG/RDF Input

- **Format:** N-Triples (`.nt`), one triple per line: `<subject> <predicate> <object> .`
- **Namespaces used:** `dcterms`, `rdfs`, `foaf`, `schema`, `fabio`, `dcat`, `mlso`, `mls`
- **Entity prefixes:**
  - Paper: `http://w3id.org/mlsea/pwc/scientificWork/`
  - Dataset: `http://w3id.org/mlsea/pwc/dataset/`
  - Model: `http://w3id.org/mlsea/pwc/model/`
- **Predicates extracted (papers):** `dcterms:title`, `rdfs:label`, `foaf:name`, `fabio:abstract`, `schema:description`, `dcterms:issued`, `dcterms:creator`, `schema:author`, `dcat:keyword`, `mlso:hasTaskType`, `mlso:hasRelatedImplementation`, `schema:codeRepository`
- **Predicates extracted (models):** additionally `mls:hasModelCharacteristic`, `mls:realizes`, `mls:hasHyperParameter`, `mlso:hasEvaluation`, `mlso:hasRun`, `mls:hasInput`, `mls:hasOutput`

### 1.5 Question Dataset

- **File:** `data/questions/ml_questions_dataset.json`
- **Total questions:** 280 (265 answerable, 15 unanswerable — determined by `is_answerable` field)
- **Schema fields:** `id`, `question`, `question_type`, `target_entity_iri`, `answer`, `answer_type`, `text_answer`, `is_answerable`
- **Gold target format:** `target_entity_iri` — an RDF IRI pointing to the correct entity
- **Question types:** include but are not limited to `paper_to_authors`, `paper_to_tasks`, `paper_to_publication_year`, `dataset_to_tasks`, `dataset_to_task_count`, model queries, cross-entity queries  
  **TODO:** Complete question-type taxonomy not explicitly documented anywhere in the repo. Must be extracted from the JSON file.
- **Difficulty levels:** `easy`, `medium`, `hard`, `unknown`  
  **TODO:** Assignment criteria not documented.

### 1.6 Chunk Generation Scripts

| Entity | Script | Representation | Max chars |
|--------|--------|----------------|-----------|
| Paper | `src/pre_retrieval/papers/chunking/build_title_only_chunks.py` | `title_only` | 512 |
| Paper | `src/pre_retrieval/papers/chunking/build_abstract_only_chunks.py` | `abstract_only` | 1,600 |
| Paper | `src/pre_retrieval/papers/chunking/build_title_abstract_chunks.py` | `title_abstract` | 1,800 |
| Paper | `src/pre_retrieval/papers/chunking/build_predicate_filtered_chunks.py` | `predicate_filtered` | 1,800 |
| Paper | `src/pre_retrieval/papers/chunking/build_enriched_paper_chunks.py` | `enriched_metadata` | 2,200 |
| Paper | `src/pre_retrieval/papers/chunking/build_one_hop_paper_chunks.py` | `one_hop` | 2,200 |
| Dataset | `src/pre_retrieval/datasets/chunking/build_dataset_*` (4 scripts) | 4 strategies | 512–2,400 |
| Model | `src/pre_retrieval/models/chunking/build_model_*` (4 scripts) | 4 strategies | 512–2,400 |

### 1.7 Embedding Scripts

- **Core logic:** `src/pre_retrieval/shared/embed_and_store.py`
- **Embedder class:** `SentenceTransformerEmbedder` in `src/pre_retrieval/shared/embedder.py`
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`; 384-dimensional vectors; L2-normalised before storage
- **Vector store:** `ChromaVectorStore` in `src/pre_retrieval/shared/vector_store.py`; HNSW index; cosine distance; score = `1.0 − cosine_distance`
- **Collection naming:** `papers_{representation_type}`, `datasets_{representation_type}`, `models_{representation_type}`
- **Batch size:** 64

### 1.8 Retrieval Scripts

| Script | Method | Description |
|--------|--------|-------------|
| `src/retrieval/dense_baseline.py` | `pure_semantic_dense` | Loads best pre-retrieval top-10; returns as-is |
| `src/retrieval/filtering.py` | `hybrid_type_filtering` | Type-based filter; control/no-op |
| `src/retrieval/filtering.py` | `hybrid_type_onehop_filtering` | Type filter + one-hop richness boost |
| `src/retrieval/filtering.py` | `hybrid_predicate_aware_filtering` | Type filter + question-type-specific predicate boost |
| `src/retrieval/rrf.py` | `optional_rrf_fusion` | Reciprocal Rank Fusion over multiple representations |
| `src/retrieval/rrf.py` | `optional_rrf_symbolic` | RRF fusion + predicate-aware filtering |

### 1.9 Evaluation Scripts

- **Pre-retrieval evaluation:** `src/pre_retrieval/shared/evaluate_retrieval.py`  
  Computes Hit@1, Hit@5, Hit@10, MRR, NDCG per question; aggregates by difficulty and category.
- **Retrieval-stage evaluation:** `src/retrieval/evaluate_retrieval_stage.py`  
  Same metrics; adds segmentation by entity_type and question_type.
- **Result aggregation:** `src/pre_retrieval/shared/aggregate_results.py`  
  Produces summary CSV/JSON/MD across all representations.
- **Metric formulas (verified in code):**
  - `hit_at_k(rank, k)` = 1 if rank < k else 0
  - `reciprocal_rank(rank)` = 1.0 / (rank + 1)
  - `ndcg(rank)` = 1.0 / log2(rank + 2) (0-indexed rank, gold contributes 1.0)

### 1.10 Result Folders

| Path | Contents |
|------|----------|
| `data/results/pre_retrieval_results/{entity_type}/{representation}/results.json` | Per-question metrics + diagnostics |
| `data/results/pre_retrieval_results/{entity_type}/{representation}/top10.json` | Top-10 retrieved candidates per question |
| `data/results/retrieval/{method_name}/results.json` | Per-question retrieval results |
| `data/results/retrieval/{method_name}/metrics.json` | Aggregated metrics (overall + by dimension) |
| `data/results/retrieval/summary.{csv,json,md}` | Cross-method comparison |
| `data/results/retrieval/thesis_tables/` | CSV tables for thesis figures |
| `data/results/thesis_tables/` | Pre-retrieval comparison tables |
| `data/results/thesis_figures/` | PDF/PNG plots |
| `data/results/summary.{csv,json,md}` | Pre-retrieval summary across all representations |

### 1.11 Implementation vs. Documentation Status

| Component | Status |
|-----------|--------|
| RDF extraction (papers, datasets, models) | **Implemented and run** |
| Chunk construction (14 representations) | **Implemented and run** |
| Dense embedding + ChromaDB indexing | **Implemented and run** |
| Pre-retrieval evaluation (Hit@k, MRR, NDCG) | **Implemented and run; results available** |
| Retrieval methods (6 methods) | **Implemented and run; results available** |
| Retrieval evaluation + segmentation | **Implemented and run; results available** |
| Post-retrieval context builder + cross-encoder | **Implemented (scaffold); not run** |
| RAG generation (TinyLlama) | **Implemented (scaffold); not run** |
| Answer evaluation (SAS, ROUGE-L, LLM judge) | **Implemented (scaffold); not run** |

**thesis_figures_tables.md is empty** — all figure/table documentation must be built from scratch in this plan.

---

## 2. Methodology Chapter Structure

### Proposed Structure

```
3. Methodology
  3.1  Methodological Overview
  3.2  Dataset and Knowledge Graph Source
  3.3  Question Set and Evaluation Design
  3.4  Pre-Retrieval Phase
       3.4.1  Motivation for Pre-Retrieval Representations
       3.4.2  RDF Extraction and Entity Record Construction
       3.4.3  Corpus Curation and Subset Selection
       3.4.4  Entity-Centric Chunk Construction
              3.4.4.1  Paper Representations
              3.4.4.2  Dataset Representations
              3.4.4.3  Model Representations
       3.4.5  Embedding Generation and Indexing
       3.4.6  Pre-Retrieval Evaluation Protocol
  3.5  Retrieval Phase
       3.5.1  Retrieval Objective and Design Rationale
       3.5.2  Dense Retrieval Baseline (pure_semantic_dense)
       3.5.3  Symbolic and Metadata-Aware Hybrid Methods
              3.5.3.1  Type-Filtered Retrieval (hybrid_type_filtering)
              3.5.3.2  One-Hop Richness-Boosted Retrieval (hybrid_type_onehop_filtering)
              3.5.3.3  Predicate-Aware Retrieval (hybrid_predicate_aware_filtering)
       3.5.4  Multi-Representation Fusion via RRF
              3.5.4.1  Pure RRF Fusion (optional_rrf_fusion)
              3.5.4.2  RRF + Symbolic Filtering (optional_rrf_symbolic)
       3.5.5  Top-k Candidate Generation
       3.5.6  Retrieval Evaluation Metrics
  3.6  Transition to Post-Retrieval
```

### Per-Section Notes

| Section | Purpose | Key Supporting Files |
|---------|---------|----------------------|
| 3.1 | Orient the reader; introduce the three-stage pipeline | `docs/post_retrieval/thesis_overview.md`, `docs/post_retrieval/CLAUDE.md` |
| 3.2 | Describe MLSea KG, N-Triples format, entity types, scale | `data/raw/pwc_1.nt`, `src/pre_retrieval/papers/raw/build_paper_records.py` |
| 3.3 | Describe question dataset, gold targets, difficulty, is_answerable | `data/questions/ml_questions_dataset.json` |
| 3.4.1 | Justify why raw RDF cannot feed an LLM directly | `docs/post_retrieval/pre_retrieval_methodology.md` |
| 3.4.2 | Explain 2-pass RDF parsing, predicates extracted, entity ID resolution | `src/pre_retrieval/papers/raw/build_paper_records.py` |
| 3.4.3 | Explain corpus curation (gold-first subset, max 200k) | `src/pre_retrieval/papers/raw/build_curated_subset.py` |
| 3.4.4 | Explain all 14 representation strategies with field contents and char limits | all `build_*_chunks.py` scripts |
| 3.4.5 | Explain SentenceTransformer, 384-dim cosine, ChromaDB, batch embedding | `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py` |
| 3.4.6 | Explain Hit@k, MRR, NDCG definitions, evaluation loop, diagnostics | `src/pre_retrieval/shared/evaluate_retrieval.py` |
| 3.5.1 | Explain transition from pre-retrieval results to retrieval design | `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/README.md` |
| 3.5.2–3.5.4 | One subsection per method family | `src/retrieval/dense_baseline.py`, `filtering.py`, `rrf.py` |
| 3.5.5 | Explain k=10 default, candidate output schema | `src/retrieval/config.py` |
| 3.5.6 | Justify NDCG as primary metric; explain others | `src/retrieval/result_schema.py`, `evaluate_retrieval_stage.py` |
| 3.6 | Bridge to Chapter 4 (post-retrieval); name what retrieval provides | `docs/post_retrieval/Post_Retrieval_Strategy.md` |

---

## 3. Pre-Retrieval Phase: Thesis Explanation Plan

### 3.1 The Problem Pre-Retrieval Solves

**What to write:**  
Standard RAG ingests text documents directly. MLSea is an RDF knowledge graph in which information about a single entity (e.g., a paper) is distributed across hundreds of individual triples. A natural-language question cannot be meaningfully matched against raw RDF triples because:

1. Triples have no lexical coherence — the subject is an IRI, the predicate is a namespace token, the object may be another IRI or a literal.
2. No single triple captures the full context needed for a relevance judgement.
3. Embedding a triple like `<pwc/scientificWork/bert> <dcterms:title> "BERT" .` produces a vector for the title string alone, losing all relational context.

**Source files:** `docs/post_retrieval/pre_retrieval_methodology.md`, `docs/post_retrieval/thesis_overview.md`

**Suggested figure:** Figure 3.1 — contrast diagram: raw RDF triple vs. assembled entity chunk (see §5).

---

### 3.2 Two-Pass RDF Extraction

**What to write:**  
The extraction pipeline (`src/pre_retrieval/papers/raw/build_paper_records.py`) performs a two-pass scan of `data/raw/pwc_1.nt` (26,606,202 N-Triple lines):

- **Pass 1:** Identify all subjects with the paper entity prefix (`http://w3id.org/mlsea/pwc/scientificWork/`) or RDF type `mlso:ScientificWork`. Collect all predicate-object pairs for each subject. Track any object URIs that are referenced by the entity (linked nodes).
- **Pass 2:** For each linked node identified in Pass 1, collect its labels and RDF types. This provides human-readable names for tasks, datasets, methods, and implementations linked to each paper.

The resulting **canonical paper record** contains:

| Field | Source Predicate(s) | Example |
|-------|---------------------|---------|
| `paper_id` | entity URI (normalised) | `pwc/scientificWork/bert` |
| `title` | `dcterms:title`, `rdfs:label`, `foaf:name` | "BERT: Pre-training..." |
| `abstract` | `fabio:abstract`, `schema:description` | — |
| `publication_year` | `dcterms:issued`, `schema:datePublished` | 2018 |
| `authors` | `dcterms:creator`, `schema:author` | ["Jacob Devlin", …] |
| `keywords` | `dcat:keyword` | ["NLP", …] |
| `tasks` | `mlso:hasTaskType` → labels | ["Question Answering", …] |
| `datasets` | linked DCAT Dataset labels | ["SQuAD", …] |
| `methods` | linked method labels | ["Transformer", …] |
| `metrics` | linked metric labels | ["F1 Score", …] |
| `implementations` | `mlso:hasRelatedImplementation`, `schema:codeRepository` | ["github.com/…"] |
| `linked_entities` | all linked nodes (predicate + label + type) | [{predicate, label, types, category}, …] |

Analogous records are built for datasets (script: `src/pre_retrieval/datasets/raw/build_dataset_records.py`) and models (script: `src/pre_retrieval/models/raw/build_model_records.py`).

**Source files to cite:** `src/pre_retrieval/papers/raw/build_paper_records.py`, `src/pre_retrieval/shared/utils.py`

**Suggested table:** Table 3.1 — RDF predicates used per entity type (see §5).

---

### 3.3 Corpus Curation

**What to write:**  
The full MLSea KG contains far more papers than can be embedded in a single experiment. A curated subset of 200,000 papers is constructed by `src/pre_retrieval/papers/raw/build_curated_subset.py` using the following rule:

1. Gold-target papers (those appearing as `target_entity_iri` in any evaluation question) are always included first.
2. Remaining capacity (up to `max_papers = 200,000`) is filled with other papers from `papers_master.jsonl`.

This ensures that every evaluation question has its correct answer present in the retrieval corpus, making the evaluation a valid closed-world test. Output: `data/intermediate/raw_papers/papers_subset_200k.jsonl`.

---

### 3.4 Representation Strategies

#### 3.4.1 Why Textual Chunks Are Necessary

The entity records contain structured data (lists of tasks, datasets, etc.). These cannot be directly fed into a vector embedding model without linearisation. Textual chunk construction converts the structured record into a natural-language-like string for each representation strategy.

#### 3.4.2 Paper Representations

Six strategies were designed and evaluated, each implemented as a separate builder script:

| Name | Content | Max chars | Script |
|------|---------|-----------|--------|
| `title_only` | Title only | 512 | `build_title_only_chunks.py` |
| `abstract_only` | Abstract only | 1,600 | `build_abstract_only_chunks.py` |
| `title_abstract` | Title + abstract | 1,800 | `build_title_abstract_chunks.py` |
| `predicate_filtered` | Title + abstract + selected metadata fields | 1,800 | `build_predicate_filtered_chunks.py` |
| `enriched_metadata` | Title + abstract (900 chars) + up to 5 tasks + 5 datasets + 5 methods + 5 metrics + 6 authors + 3 implementations | 2,200 | `build_enriched_paper_chunks.py` |
| `one_hop` | Title + abstract (700 chars) + up to 12 linked entities grouped by category (tasks, datasets, methods, metrics, implementations) | 2,200 | `build_one_hop_paper_chunks.py` |

**Key distinctions:**
- `enriched_metadata` explicitly enumerates ML-domain fields (tasks, datasets, methods, metrics). It is the most semantically specific representation.
- `one_hop` uses the linked-entity graph structure: it groups all RDF-linked nodes by their inferred category, providing broader relational context but at the cost of added noise.
- `predicate_filtered` lies between: it selects a whitelist of predicates rather than using all linked entities.
- `abstract_only` strips the title, which — as the pre-retrieval results show — severely reduces retrieval performance.

**Source files:** `src/pre_retrieval/papers/chunking/build_enriched_paper_chunks.py`, `build_one_hop_paper_chunks.py`, `build_representations.py`

**Suggested figure:** Figure 3.2 — side-by-side example of the same paper in `title_only`, `enriched_metadata`, and `one_hop` formats (see §5).

#### 3.4.3 Dataset Representations

| Name | Content |
|------|---------|
| `dataset_title_only` | Title only |
| `dataset_metadata` | Title + description + tasks + related papers |
| `dataset_predicate_filtered` | Selected predicate subset |
| `dataset_enriched_metadata` | Title + description + related papers + tasks + implementations + linked entities |

**Key observation:** Dataset entities in MLSea are sparsely annotated. Many datasets have only a title and minimal metadata, making richer representations counterproductive (they include more empty or noisy fields). `dataset_title_only` is the best representation (NDCG 0.3822) — see §3.6 pre-retrieval results.

#### 3.4.4 Model Representations

| Name | Content |
|------|---------|
| `model_title_only` | Title only |
| `model_metadata` | Title + description + tasks + related data |
| `model_predicate_filtered` | Filtered predicates (tasks, datasets, related papers) |
| `model_enriched_metadata` | Title + all enriched metadata + linked entities |

**Key observation:** Models benefit from predicate-filtered representations. Graph-aware filtering (`model_predicate_filtered`) achieves NDCG 0.8750 — the highest of all 14 representations — because ML models are most distinctively described by their task and dataset associations.

---

### 3.5 Embedding Generation

**What to write:**

All text representations are embedded using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace SentenceTransformers), producing 384-dimensional dense vectors. Embeddings are L2-normalised before storage.

The choice of `all-MiniLM-L6-v2` balances:
- Embedding quality sufficient for semantic search (competitive on MTEB benchmarks)
- Inference efficiency (6-layer distilled model, ~23M parameters)
- Availability and reproducibility (openly available, no API key required)

Each representation type is stored in a separate Chroma collection, named `papers_{representation_type}` (or `datasets_`, `models_`). The vector store uses a HNSW index with cosine distance as the similarity metric. At query time, the score returned to the evaluation pipeline is computed as `score = 1.0 − cosine_distance`, so higher scores indicate greater similarity.

Questions are embedded using the same model and the same normalization procedure. The system embeds all 280 evaluation questions and queries the Chroma collection for top-10 nearest neighbors.

**Source files:** `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py`, `config/pre_retrieval_config.json`

---

### 3.6 Pre-Retrieval Evaluation Protocol

**What to write:**

The evaluation measures how well each text representation strategy supports semantic retrieval of the correct entity for a given question.

**Gold target resolution:** Each question has a `target_entity_iri`. The evaluation pipeline checks whether this IRI corresponds to an entity present in the Chroma collection. If the entity is missing (e.g., falls outside the 200k subset for non-gold entities), the question is marked as unmatched and excluded.

**Evaluable questions:** Of 280 total questions, 265 are marked `is_answerable = true` and have matching entities in the collection. The 15 unanswerable questions are retained in the dataset but excluded from metric averaging.

**Metrics computed** (implemented in `src/pre_retrieval/shared/evaluate_retrieval.py`):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Hit@1 | 1 if gold at rank 1 else 0 | Precision of the top result |
| Hit@5 | 1 if gold in top 5 else 0 | Lenient precision |
| Hit@10 | 1 if gold in top 10 else 0 | Recall upper bound |
| MRR | 1 / (rank + 1) | Weighted average precision |
| NDCG | 1 / log₂(rank + 2) | Discounted precision; primary metric |

Results are segmented by difficulty level and by question category for deeper analysis.

**Key pre-retrieval results (verified from `data/results/summary.md` and `data/results/pre_retrieval_results/`):**

| Entity type | Representation | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG |
|-------------|---------------|-------|-------|--------|-----|------|
| Paper | enriched_metadata | 0.7753 | 0.8483 | 0.8539 | 0.8117 | **0.8225** |
| Paper | predicate_filtered | 0.7247 | 0.8090 | 0.8202 | 0.7596 | 0.7745 |
| Paper | one_hop | 0.7135 | — | — | — | 0.7642 |
| Paper | title_only | 0.7022 | — | — | — | 0.7346 |
| Paper | title_abstract | 0.6236 | — | — | — | 0.6934 |
| Paper | abstract_only | 0.4213 | — | — | — | 0.5438 |
| Dataset | dataset_title_only | 0.2807 | — | — | — | **0.3822** |
| Dataset | dataset_enriched_metadata | 0.1579 | 0.4211 | 0.5263 | 0.2613 | 0.3243 |
| Dataset | dataset_metadata | 0.1404 | — | — | — | 0.2657 |
| Dataset | dataset_predicate_filtered | 0.0877 | — | — | — | 0.1919 |
| Model | model_predicate_filtered | 0.8000 | 0.9000 | 0.9333 | 0.8556 | **0.8750** |
| Model | model_enriched_metadata | 0.6000 | — | — | — | 0.6916 |
| Model | model_metadata | 0.3333 | — | — | — | 0.4733 |
| Model | model_title_only | 0.3667 | — | — | — | 0.4465 |

**Main finding:** There is no universal best representation across entity types. Papers require enriched semantic metadata; models require graph-aware predicate filtering; datasets are best served by minimal title-only representations due to sparse annotation in the KG.

**Suggested tables:** Table 3.2 (full pre-retrieval comparison) and Table 3.3 (best per entity type) — see §5.

---

## 4. Retrieval Phase: Thesis Explanation Plan

### 4.1 Retrieval Objective

**What to write:**

The retrieval stage addresses a different research question than pre-retrieval: rather than asking *which text representation is optimal*, it asks *which candidate generation strategy produces the best ranked list of candidates* given a fixed, optimal representation.

The best pre-retrieval representation per entity type is used as the input to all retrieval methods:
- Papers: `enriched_metadata` (NDCG 0.8225)
- Datasets: `dataset_title_only` (NDCG 0.3822)
- Models: `model_predicate_filtered` (NDCG 0.8750)

The pre-retrieval top-10 candidates (`top10.json`) serve as the candidate pool. The retrieval stage re-orders, filters, or fuses these candidates.

**Key boundary:** Re-ranking using a cross-encoder is explicitly *not* part of the retrieval stage. It belongs to post-retrieval. The retrieval stage produces a ranked list of top-10 candidates per question.

**Source files:** `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/README.md`, `src/retrieval/config.py`

---

### 4.2 Method Family 1 — Dense Retrieval Baseline (`pure_semantic_dense`)

**Input:** Pre-retrieval top-10 list for the best representation per entity type  
**Algorithm:** Direct passthrough — the semantic ranking from the pre-retrieval stage is used without modification  
**Output:** Top-10 candidates ordered by cosine similarity score  
**Implementation:** `src/retrieval/dense_baseline.py`  

**Expected strengths:** High semantic precision; robust for queries where vocabulary directly matches representation content  
**Expected weaknesses:** Ignores KG graph structure; treats all entity types uniformly; no symbolic signal  

**Thesis description:** This method establishes the baseline. Any retrieval method that does not outperform it fails to justify its added complexity.

**Results (verified from `data/results/retrieval/summary.md`):**
- NDCG: 0.7337, Hit@1: 0.6717, Hit@5: TODO (fill from detailed metrics), MRR: TODO, evaluated on 265 questions

---

### 4.3 Method Family 2 — Semantic + Symbolic Hybrid Methods

#### 4.3.1 `hybrid_type_filtering`

**Input:** Pre-retrieval top-10 list  
**Algorithm:**  
1. Determine expected entity type from `target_entity_iri` prefix (paper / dataset / model)
2. Filter candidates to retain only those matching the expected type
3. Fallback: if no candidates survive, return original unfiltered list  
**Output:** Type-filtered top-10 list in original semantic order  
**Implementation:** `src/retrieval/filtering.py`

**Design rationale:** In the pre-retrieval stage, each Chroma collection is already restricted to a single entity type (one collection per representation-type). This method is therefore a control: it confirms that the collection is pure and type-filtering adds no new information. The fact that `hybrid_type_filtering` produces identical results to `pure_semantic_dense` confirms this (NDCG 0.7337, Hit@1 0.6717 — same values).

**Expected strengths:** Zero-noise operation; confirms system correctness  
**Expected weaknesses:** No-op for the current system architecture  

#### 4.3.2 `hybrid_type_onehop_filtering`

**Input:** Pre-retrieval top-10 list  
**Algorithm:**  
1. Type filter (same as above)
2. Score each candidate by one-hop richness: count non-empty fields among {tasks, datasets, methods, metrics, implementations} + bonus if `"Linked Entities"` appears in source text
3. Re-sort by richness score (descending)  
**Output:** Type-filtered, richness-re-ranked top-10  
**Implementation:** `src/retrieval/filtering.py` — function `_onehop_richness()`

**Design rationale:** Graph-richer entities are hypothesised to be more informative for multi-field questions. This method tests whether surface metadata density correlates with retrieval quality.

**Expected strengths:** May help for questions requiring multiple facts (e.g., "What tasks does this paper address?")  
**Expected weaknesses:** Richness is a proxy signal; a well-described but irrelevant entity outranks a sparse but correct one  

**Results:** NDCG 0.7351, Hit@1 0.6717 — marginal improvement in NDCG, no change in Hit@1.

#### 4.3.3 `hybrid_predicate_aware_filtering`

**Input:** Pre-retrieval top-10 list  
**Algorithm:**  
1. Type filter  
2. Map `question_type` to a target metadata field:
   - `paper_to_tasks` / task-related → boost candidates with non-empty `tasks`
   - `paper_to_implementation` → boost candidates with non-empty `implementations`
   - `paper_to_publication_year` → boost candidates with non-empty `year`
   - `repository` / `model_family` → boost candidates mentioning "Linked Entities"
3. Split candidates into boosted (matching field) + remainder; concatenate  
4. Fallback: if no evidence found, return original order  
**Output:** Predicate-aware re-ranked top-10  
**Implementation:** `src/retrieval/filtering.py` — function `_boost_by_predicate()`

**Design rationale:** Different question types interrogate different KG predicates. A question about tasks should prefer candidates that explicitly list tasks. This method encodes question-type-to-predicate mappings as a symbolic overlay.

**Expected strengths:** Targeted precision improvement for specific question types; best MRR of all methods (0.7233)  
**Expected weaknesses:** Mapping must be manually defined; fails for unseen question types  

**Results:** NDCG 0.7375, Hit@1 0.6868, MRR 0.7233 — best precision among hybrid methods.

---

### 4.4 Method Family 3 — Multi-Representation Fusion via RRF

#### 4.4.1 `optional_rrf_fusion`

**Input:** Pre-retrieval top-10 lists for *multiple* representations per entity type  
**Fusion groups (from `src/retrieval/config.py`):**
- Paper: [`enriched_metadata`, `predicate_filtered`, `one_hop`]
- Dataset: [`dataset_title_only`, `dataset_enriched_metadata`]
- Model: [`model_predicate_filtered`, `model_enriched_metadata`]

**Algorithm:**  
Reciprocal Rank Fusion (RRF):
```
score(d) = Σ_i  1 / (k + rank_i(d))     k = 60
```
For each entity, aggregate scores across all representations. Sort by descending aggregated score.

**Output:** RRF-fused top-10  
**Implementation:** `src/retrieval/rrf.py`

**Design rationale:** A document ranked first in one representation but absent in another will still receive a positive score. RRF is robust to outlier rankings and does not require score calibration across representations.

**Expected strengths:** Broader recall (improves Hit@10); less sensitive to single-representation failures  
**Expected weaknesses:** May depress Hit@1 if conflicting representations disagree on the top candidate  

**Results:** NDCG 0.7354, Hit@1 0.6491 — RRF improves Hit@5/Hit@10 but lowers Hit@1 vs. baseline.

#### 4.4.2 `optional_rrf_symbolic`

**Input:** Same as `optional_rrf_fusion`  
**Algorithm:** RRF fusion → then apply `hybrid_predicate_aware_filtering` on fused list  
**Output:** RRF-fused, predicate-re-ranked top-10  
**Implementation:** `src/retrieval/rrf.py`

**Expected strengths:** Combines breadth (RRF) with precision (predicate boosting)  
**Expected weaknesses:** Additive complexity; symbolic boosting may partially undo RRF ranking  

**Results:** NDCG 0.7434, Hit@1 0.6642 — highest NDCG of all methods, but note tight clustering.

---

### 4.5 Top-k Candidate Generation

- Default k = 10 (configured in `src/retrieval/config.py`)
- Each method outputs a `results.json` with per-question candidate lists and `metrics.json` with aggregated metrics
- Top-10 candidate list is the handoff to post-retrieval; each candidate carries: `item_id`, `title`, `score`, `representation_type`, and `source_text`

---

### 4.6 Retrieval Evaluation Metrics

**Why NDCG is the primary metric:**  
NDCG discounts the contribution of the correct answer by its rank position. A correct answer at rank 1 contributes 1.0 / log₂(2) = 1.0; at rank 5 it contributes 1.0 / log₂(6) ≈ 0.39. This is appropriate for RAG because the highest-ranked candidates are most likely to be used in context construction.

**Why MRR is secondary:**  
MRR = 1 / rank focuses on the single highest-ranked relevant item, useful for single-answer questions.

**Why Hit@k is complementary:**  
Hit@10 provides an upper bound on the recall achievable by any downstream reranker. If the gold entity is not in the top-10, no post-retrieval method can recover it.

**Why Precision@k is not used:**  
Each question has exactly one gold entity. Precision@k is identical to Hit@k in this setting and adds no information.

**Segmentation dimensions (verified in `src/retrieval/evaluate_retrieval_stage.py`):**
- `by_difficulty`: easy / medium / hard / unknown
- `by_entity_type`: paper / dataset / model
- `by_question_type`: various (see question_type field in questions JSON)

**Consolidated retrieval results (verified from `data/results/retrieval/summary.md`):**

| Method | Evaluated | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG |
|--------|-----------|-------|-------|--------|-----|------|
| `optional_rrf_symbolic` | 265 | 0.6642 | — | — | — | **0.7434** |
| `hybrid_predicate_aware_filtering` | 265 | **0.6868** | — | — | **0.7233** | 0.7375 |
| `optional_rrf_fusion` | 265 | 0.6491 | — | — | — | 0.7354 |
| `hybrid_type_onehop_filtering` | 265 | 0.6717 | — | — | — | 0.7351 |
| `pure_semantic_dense` | 265 | 0.6717 | — | — | — | 0.7337 |
| `hybrid_type_filtering` | 265 | 0.6717 | — | — | — | 0.7337 |

**TODO:** Fill Hit@5, Hit@10, MRR for all methods from `data/results/retrieval/{method}/metrics.json`.

**Key observations:**
1. All methods cluster in a narrow NDCG band (0.7337–0.7434). The improvement over the dense baseline is at most +0.0097 NDCG points.
2. `hybrid_type_filtering` ≡ `pure_semantic_dense` (identical results), confirming per-type collections are already pure.
3. `hybrid_predicate_aware_filtering` achieves the best Hit@1 and MRR, suggesting it is most effective at placing the correct answer at rank 1.
4. RRF methods trade Hit@1 for recall breadth (higher Hit@5/Hit@10).
5. The tight clustering suggests that the semantic representation quality (pre-retrieval) is the binding constraint, and symbolic overlays provide marginal gains at this stage.

---

### 4.7 Transition to Post-Retrieval

The retrieval stage delivers a ranked list of top-10 candidates per question. Post-retrieval consumes these lists and:

1. Resolves full canonical records (title, abstract, authors, metadata) from `papers_master.jsonl`
2. Applies score-based hard filtering (threshold: cosine score > 0.20)
3. Re-ranks using a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
4. Formats the top candidates into an LLM prompt context
5. Generates an answer using a language model
6. Evaluates the answer with SAS, ROUGE-L, and an LLM-as-a-Judge score

**Source files:** `docs/post_retrieval/Post_Retrieval_Strategy.md`, `src/post_retrieval/pipeline/context_builder.py`

---

## 5. Figures and Tables Plan

> **Note:** `docs/post_retrieval/thesis_figures_tables.md` is currently empty. The items below constitute its complete content.

### Figures

---

**Figure 3.1 — Full RAG Pipeline Overview**  
- **Section:** 3.1 Methodological Overview  
- **What it shows:** Three-stage pipeline diagram: KG Input → Pre-Retrieval (extraction, chunking, embedding) → Retrieval (6 methods) → Post-Retrieval (re-ranking, generation, evaluation)  
- **Introduces:** "Figure 3.1 illustrates the overall architecture of the proposed KG-RAG system."  
- **Follows:** "The pipeline is described in detail in the following sections."  
- **Status:** Must be created — draw as a flow diagram (PowerPoint, draw.io, or TikZ)  
- **Likely path:** `data/results/thesis_figures/pipeline_overview.pdf`

---

**Figure 3.2 — KG-to-Chunk Transformation**  
- **Section:** 3.4.1 Motivation for Pre-Retrieval Representations  
- **What it shows:** Left side: three raw RDF triples for a single paper entity (IRI, predicate, object). Right side: the same paper as an `enriched_metadata` text chunk.  
- **Introduces:** "Figure 3.2 contrasts a raw RDF representation with its linearised chunk equivalent."  
- **Follows:** "The transformation is necessary because embedding models operate over natural-language strings, not graph triples."  
- **Status:** Must be created — use a concrete example from `data/intermediate/representations/papers/enriched_metadata.jsonl`  
- **Likely path:** `data/results/thesis_figures/kg_to_chunk.pdf`

---

**Figure 3.3 — Representation Strategy Comparison (Paper)**  
- **Section:** 3.4.4.1 Paper Representations  
- **What it shows:** Side-by-side text boxes showing the same paper rendered in `title_only`, `enriched_metadata`, and `one_hop` formats  
- **Introduces:** "Figure 3.3 shows the same scientific publication rendered under three representation strategies."  
- **Follows:** "The strategies differ in which facets of the entity are included and to what depth linked entities are traversed."  
- **Status:** Must be created — draw from a single example record  
- **Likely path:** `data/results/thesis_figures/representation_examples.pdf`

---

**Figure 3.4 — NDCG by Representation (Pre-Retrieval, All Entities)**  
- **Section:** 3.4.6 Pre-Retrieval Evaluation Protocol  
- **What it shows:** Grouped bar chart: x-axis = representation name; y-axis = NDCG; bars grouped by entity type (paper/dataset/model); clear visual of enriched_metadata winning for papers, predicate_filtered for models, title_only for datasets  
- **Introduces:** "Figure 3.4 compares NDCG scores across all 14 representation strategies."  
- **Follows:** "The results confirm that no single representation dominates across entity types."  
- **Status:** Partially exists — check `data/results/thesis_figures/ndcg_*.pdf`; regenerate as grouped comparison  
- **Likely path:** `data/results/thesis_figures/ndcg_all_representations.pdf`

---

**Figure 3.5 — Retrieval Workflow Diagram**  
- **Section:** 3.5.1 Retrieval Objective  
- **What it shows:** Flow: question → embed → query Chroma (best representation) → top-10 candidates → apply retrieval method → re-ranked top-10 → metrics evaluation  
- **Introduces:** "Figure 3.5 depicts the retrieval workflow applied to each of the six evaluated methods."  
- **Status:** Must be created  
- **Likely path:** `data/results/thesis_figures/retrieval_workflow.pdf`

---

**Figure 3.6 — Retrieval Method Comparison (NDCG)**  
- **Section:** 3.5.6 Retrieval Evaluation Metrics  
- **What it shows:** Horizontal bar chart of NDCG values for all 6 methods; tight clustering (0.7337–0.7434) visible  
- **Introduces:** "Figure 3.6 summarises the NDCG performance of all six retrieval methods evaluated on 265 questions."  
- **Follows:** "The narrow performance band suggests that semantic representation quality, not retrieval strategy, is the binding constraint at this stage."  
- **Status:** Data available in `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`; figure must be generated  
- **Likely path:** `data/results/thesis_figures/retrieval_method_comparison.pdf`

---

**Figure 3.7 — Retrieval NDCG by Entity Type**  
- **Section:** 3.5.6 Retrieval Evaluation Metrics  
- **What it shows:** Grouped bar chart: x-axis = method; bars grouped by entity type; reveals which methods help most for datasets vs. papers vs. models  
- **Status:** Data available in `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv`  
- **Likely path:** `data/results/thesis_figures/retrieval_by_entity_type.pdf`

---

**Figure 3.8 — Pre-Retrieval to Post-Retrieval Pipeline Bridge**  
- **Section:** 3.6 Transition to Post-Retrieval  
- **What it shows:** Handoff diagram: top-10 retrieval output (candidates + scores) → post-retrieval context builder → cross-encoder re-ranking → LLM prompt  
- **Status:** Must be created  
- **Likely path:** `data/results/thesis_figures/post_retrieval_bridge.pdf`

---

### Tables

---

**Table 3.1 — RDF Predicates Used Per Entity Type**  
- **Section:** 3.4.2 RDF Extraction  
- **Columns:** Entity type | Predicate URI | Mapped field | Example value  
- **Introduces:** "Table 3.1 lists the RDF predicates extracted from MLSea for each entity type."  
- **Status:** Must be compiled manually from `src/pre_retrieval/papers/raw/build_paper_records.py` and equivalents  

---

**Table 3.2 — Full Pre-Retrieval Representation Comparison**  
- **Section:** 3.4.6 Pre-Retrieval Evaluation Protocol  
- **Columns:** Entity type | Representation | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG  
- **Data source:** `data/results/thesis_tables/full_comparison.csv` (verified to exist)  
- **Introduces:** "Table 3.2 reports evaluation metrics for all 14 representation strategies across 265 answerable questions."  
- **Follows:** Discussion of entity-specific results.  
- **Status:** Data exists in `data/results/thesis_tables/full_comparison.csv`. Table needs formatting for LaTeX/Word.  
- **Caption suggestion:** "Table 3.2: Pre-retrieval evaluation metrics (Hit@1, Hit@5, Hit@10, MRR, NDCG) for all 14 representation strategies evaluated on 265 answerable questions from the ML question dataset. Best result per entity type highlighted in bold."

---

**Table 3.3 — Best Representation Per Entity Type**  
- **Section:** 3.4.6 Pre-Retrieval Evaluation Protocol  
- **Columns:** Entity type | Best representation | Hit@1 | MRR | NDCG | Notes  
- **Data source:** `data/results/thesis_tables/best_per_entity.csv` (verified to exist)  
- **Status:** Data exists. Needs LaTeX/Word formatting.

---

**Table 3.4 — Retrieval Method Comparison**  
- **Section:** 3.5.6 Retrieval Evaluation Metrics  
- **Columns:** Method | Evaluated questions | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG  
- **Data source:** `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` (verified to exist), `data/results/retrieval/summary.md`  
- **Caption suggestion:** "Table 3.4: Retrieval stage evaluation metrics for all six retrieval methods evaluated on 265 answerable questions. Primary metric is NDCG. Δ vs. `pure_semantic_dense` baseline shown in parentheses."  
- **Status:** Data exists; needs LaTeX/Word formatting and delta column.

---

**Table 3.5 — Retrieval NDCG by Question Difficulty**  
- **Section:** 3.5.6 Retrieval Evaluation Metrics  
- **Columns:** Method | Easy NDCG | Medium NDCG | Hard NDCG  
- **Data source:** `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv`  
- **Status:** Data exists.

---

**Table 3.6 — Retrieval NDCG by Entity Type**  
- **Section:** 3.5.6 Retrieval Evaluation Metrics  
- **Columns:** Method | Paper NDCG | Dataset NDCG | Model NDCG  
- **Data source:** `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv`  
- **Status:** Data exists.

---

**Table 3.7 — Pre-Retrieval Metric Definitions**  
- **Section:** 3.4.6 Pre-Retrieval Evaluation Protocol  
- **Columns:** Metric | Formula | Range | Primary use  
- **Status:** Must be written manually (trivial; no code needed).  

---

## 6. Thesis-Ready Writing Blocks

> These paragraphs are first-draft quality. Numbers and citations are verified from repository files. Sections marked TODO require additional data collection before finalisation.

---

### Block 6.1 — Methodological Overview (§3.1)

This chapter describes the design and implementation of a Retrieval-Augmented Generation (RAG) pipeline for machine-learning-specific question answering over the MLSea knowledge graph. The pipeline is divided into three stages: pre-retrieval, retrieval, and post-retrieval. In the pre-retrieval stage, entities from the MLSea RDF graph are extracted and converted into textual chunk representations, which are then embedded into a dense vector index. In the retrieval stage, a natural-language question is embedded and compared against the indexed representations to produce a ranked list of candidate entities. In the post-retrieval stage, the top candidates are re-ranked by a cross-encoder model, assembled into a generation context, and fed to a language model to produce a final answer. The present chapter focuses on the pre-retrieval and retrieval stages; the post-retrieval stage is introduced at the end of this chapter and elaborated in Chapter 4.

The design of the pipeline is motivated by a fundamental structural property of RDF knowledge graphs: entity information is distributed across many individual triples rather than concentrated in a single retrievable unit. Conventional dense retrieval systems assume that the corpus consists of coherent text passages. In the MLSea graph, this assumption does not hold: a query such as "What tasks does BERT address?" cannot be answered by matching against a single triple, because the answer is encoded in a set of linked nodes connected by `mlso:hasTaskType` predicates. The pre-retrieval stage addresses this structural mismatch by constructing entity-centric textual representations that consolidate the distributed graph information into a single retrievable string.

---

### Block 6.2 — Knowledge Graph and Dataset Description (§3.2)

The primary data source for this thesis is the MLSea knowledge graph, an RDF-based export of the Papers with Code repository. The graph is distributed as a single N-Triples file (`data/raw/pwc_1.nt`) comprising 26,606,202 triples and occupying 6.4 gigabytes on disk. MLSea encodes three primary entity types: scientific publications (referred to as *papers*), machine learning datasets, and machine learning models. Each entity is identified by a unique IRI following the pattern `http://w3id.org/mlsea/pwc/{type}/{identifier}`, where `{type}` is one of `scientificWork`, `dataset`, or `model`.

Entity attributes are encoded using a heterogeneous set of RDF predicates drawn from multiple vocabularies, including Dublin Core Terms (`dcterms`), the Data Catalog Vocabulary (`dcat`), the Machine Learning Schema (`mls`), the Machine Learning Schema Ontology (`mlso`), FOAF, and schema.org. Papers are annotated with titles, abstracts, publication years, author lists, task types, linked datasets, methods, metrics, and implementation pointers. Datasets are annotated with titles, descriptions, publication years, keywords, and related papers. Models are annotated with titles, descriptions, tasks, evaluation runs, and hyperparameter information.

An important characteristic of the graph is the high variability in annotation density across entities. While high-profile papers in the corpus may have dozens of linked task, dataset, and method nodes, less prominent papers — and the majority of dataset and model entities — are sparsely annotated. This sparsity has direct consequences for the quality of textual representations constructed in the pre-retrieval stage (Section 3.4).

---

### Block 6.3 — Question Set and Evaluation Design (§3.3)

The evaluation framework is built around a manually curated question dataset comprising 280 natural-language questions drawn from the ML domain (`data/questions/ml_questions_dataset.json`). Each question is annotated with: a unique identifier, the question text, a `question_type` label, a `target_entity_iri` pointing to the gold answer entity in MLSea, a reference answer, a `text_answer` field containing a human-readable gold answer string, a binary `is_answerable` flag, and a difficulty level.

Of the 280 questions, 265 are marked as answerable (`is_answerable = true`) and are used for all metric computations. The remaining 15 unanswerable questions are retained in the dataset but excluded from averaging to avoid inflating error rates. Questions span multiple `question_type` categories — including `paper_to_authors`, `paper_to_tasks`, `paper_to_publication_year`, `dataset_to_tasks`, and cross-entity queries — and are labelled by difficulty as easy, medium, or hard.

The gold-target entity IRI (`target_entity_iri`) is used as the ground truth for all retrieval evaluation. The evaluation pipeline normalises both the gold IRI and all retrieved candidate IRIs (URL-decoding and stripping graph-database wrapper prefixes) before comparison, ensuring that formatting artefacts do not cause false negatives.

---

### Block 6.4 — Motivation for Pre-Retrieval (§3.4.1)

Standard retrieval-augmented generation assumes that the knowledge corpus is composed of coherent natural-language passages. In such a setting, a dense embedding of the passage naturally captures the semantic content that may be relevant to an incoming query. The MLSea knowledge graph does not satisfy this assumption. The information associated with a single entity, such as a research paper, is distributed across hundreds of individual RDF triples, each encoding a single predicate-object relationship. A triple such as `<pwc/scientificWork/bert> <mlso:hasTaskType> <pwc/task/question-answering>` encodes a meaningful fact, but in isolation it provides no context for a similarity computation with a question like "What question answering papers use BERT?"

Moreover, the object of such a triple is itself an IRI — not a human-readable label — requiring a second graph traversal to resolve to the string "Question Answering". This two-level indirection means that simple triple-level embedding approaches would embed IRIs rather than concepts, producing uninformative vectors. The pre-retrieval stage resolves both problems by constructing *entity-centric chunk representations*: single strings that consolidate all relevant information about an entity into a format suitable for dense embedding.

---

### Block 6.5 — Chunk Construction (§3.4.4)

For each entity type, a set of representation strategies is designed to explore the trade-off between representation specificity, information density, and noise. All strategies produce a flat UTF-8 string from the structured entity record; the differences lie in which fields are included, how they are concatenated, and how deeply linked graph nodes are traversed.

For papers, six strategies are implemented. The simplest strategy, `title_only`, includes only the paper title (up to 512 characters). The `abstract_only` strategy includes only the abstract (up to 1,600 characters), deliberately omitting the title to test whether the abstract alone carries sufficient retrieval signal. The `title_abstract` strategy combines both fields (1,800 characters total). The `predicate_filtered` strategy adds a curated selection of structured metadata fields (1,800 characters). The `enriched_metadata` strategy produces the most information-dense representation: it concatenates the title, a truncated abstract (900 characters), up to five tasks, five linked datasets, five methods, five metrics, six authors, and three implementation pointers, totalling up to 2,200 characters. The `one_hop` strategy takes a graph-centric approach: it includes the title, a shorter abstract (700 characters), and up to twelve linked entities grouped by inferred category (tasks, datasets, methods, metrics, implementations), also up to 2,200 characters.

Dataset representations follow the same rationale but are constrained to four strategies due to the sparser annotation in MLSea: `dataset_title_only`, `dataset_metadata`, `dataset_predicate_filtered`, and `dataset_enriched_metadata`. Model representations are similarly structured into four strategies: `model_title_only`, `model_metadata`, `model_predicate_filtered`, and `model_enriched_metadata`.

All chunks are stored as JSONL files under `data/intermediate/representations/{entity_type}/{representation_name}.jsonl`. Each record carries the chunk text alongside the entity identifier, representation type, and text-length statistics.

---

### Block 6.6 — Embedding Generation (§3.4.5)

All textual representations are encoded using the `sentence-transformers/all-MiniLM-L6-v2` SentenceTransformer model, which produces 384-dimensional dense vectors (`src/pre_retrieval/shared/embedder.py`). Embeddings are L2-normalised before storage. The model was selected for its balance of encoding quality, inference speed, and open availability.

Encoded representations are stored in a ChromaDB vector store (`data/intermediate/chroma/`) using an HNSW index with cosine distance as the similarity metric. Each of the 14 representation strategies is stored in a separate Chroma collection, allowing independent retrieval experiments per strategy. The vector store occupies 8.2 gigabytes and contains 18 collections (14 representation collections plus auxiliary collections). At query time, the natural-language question is embedded with the same model and used to query the relevant collection. The retrieval score is derived as `score = 1.0 − cosine_distance`, yielding values in [0, 1] where 1 indicates identical vectors.

---

### Block 6.7 — Pre-Retrieval Evaluation (§3.4.6)

The quality of each representation strategy is evaluated by the degree to which the dense embedding of a question retrieves the correct gold entity within the top-10 candidates. Five metrics are computed: Hit@1, Hit@5, Hit@10, MRR, and NDCG. NDCG is designated as the primary metric because it penalises lower-ranked correct answers proportionally to their rank position, reflecting the importance of top-ranked candidates in downstream context construction.

Evaluation is performed over 265 answerable questions. For each question, the pipeline embeds the question, queries the Chroma collection for the representation under evaluation, and checks whether the gold entity IRI appears in the top-10 results. Results are aggregated overall and segmented by difficulty level and question category.

The best pre-retrieval representation for papers is `enriched_metadata` (NDCG 0.8225, Hit@1 0.7753), for models is `model_predicate_filtered` (NDCG 0.8750, Hit@1 0.8000), and for datasets is `dataset_title_only` (NDCG 0.3822, Hit@1 0.2807). The entity-type-specific variation in optimal strategy constitutes the primary finding of the pre-retrieval stage: there is no universal best representation across entity types. Papers benefit from enriched semantic metadata because their linked tasks and datasets provide distinctive vocabulary that aligns with question phrasing. Models benefit from graph-aware predicate filtering because their most distinctive attributes — task associations and linked datasets — are precisely captured by the filtered predicates. Datasets, by contrast, are sparsely annotated in MLSea, and richer representations introduce noise rather than signal; the title alone is the most reliable descriptor.

---

### Block 6.8 — Retrieval Methodology (§3.5.1)

The retrieval stage takes as its starting point the best pre-retrieval representation per entity type and evaluates whether alternative candidate generation strategies can improve the ranked output. The input to each retrieval method is the pre-computed top-10 candidate list from the pre-retrieval stage (`data/results/pre_retrieval_results/{entity_type}/{representation}/top10.json`). Six methods are evaluated, organised into three families.

The first family comprises a single method, `pure_semantic_dense`, which uses the pre-retrieval semantic ranking directly as the retrieval output. This establishes the baseline. The second family comprises three hybrid methods that overlay symbolic signals from the KG metadata onto the semantic ranking: `hybrid_type_filtering` (a control method confirming collection purity), `hybrid_type_onehop_filtering` (re-ranking by graph connectivity richness), and `hybrid_predicate_aware_filtering` (re-ranking by question-type-specific predicate presence). The third family comprises two Reciprocal Rank Fusion methods that aggregate rankings from multiple representation strategies: `optional_rrf_fusion` and `optional_rrf_symbolic`.

All six methods are evaluated on 265 answerable questions using Hit@1, Hit@5, Hit@10, MRR, and NDCG, with additional segmentation by difficulty, entity type, and question type.

---

### Block 6.9 — Retrieval Evaluation (§3.5.6)

The six retrieval methods produce NDCG values ranging from 0.7337 to 0.7434 across 265 evaluated questions. The `pure_semantic_dense` baseline achieves NDCG 0.7337 and Hit@1 0.6717. The `hybrid_type_filtering` method produces identical results (NDCG 0.7337, Hit@1 0.6717), confirming that the entity-type collections maintained by the pre-retrieval stage are already pure and require no additional type filtering.

Among hybrid methods, `hybrid_predicate_aware_filtering` achieves the best Hit@1 (0.6868) and MRR (0.7233), outperforming the baseline on precision-oriented metrics. The RRF methods improve NDCG at the cost of Hit@1: `optional_rrf_symbolic` achieves the highest NDCG (0.7434) but a lower Hit@1 (0.6642) than the baseline, reflecting the recall-broadening effect of fusing multiple representation rankings.

The narrow spread across methods — a range of only 0.0097 NDCG points — indicates that the binding constraint is not the retrieval strategy but the quality of the semantic representation produced in the pre-retrieval stage. This result is consistent with the known behaviour of dense retrieval systems: once the representation captures the relevant semantic content, the ranking is largely determined by the embedding geometry, and symbolic post-processing yields diminishing returns. It further motivates the post-retrieval re-ranking stage, which applies a cross-encoder model to re-evaluate candidate relevance at higher computational cost.

---

### Block 6.10 — Transition to Post-Retrieval (§3.6)

The retrieval stage delivers, for each evaluation question, a ranked list of up to ten candidate entities. Each candidate is characterised by its entity identifier, its text representation, and a retrieval score derived from cosine similarity. This output serves as the input to the post-retrieval stage, which is described in Chapter 4.

The post-retrieval stage performs three operations not available to the retrieval stage: (1) score-based hard filtering, which eliminates candidates with cosine scores below 0.20; (2) cross-encoder re-ranking, which scores each candidate independently against the full question using a `cross-encoder/ms-marco-MiniLM-L-6-v2` model, trading retrieval speed for greater precision; and (3) context assembly, which formats the top-ranked candidates as a structured prompt context for the language model. The quality of the retrieval output directly bounds the achievable quality of the post-retrieval stage: if the gold entity is not present in the top-10 retrieval output, it cannot be recovered by re-ranking.

---

## 7. Missing Information Checklist

### 7.1 Missing Metric Values

- [ ] **Hit@5 and Hit@10 for retrieval methods:** `data/results/retrieval/{method}/metrics.json` exists but values not extracted in this plan. Read each file and populate Table 3.4.
- [ ] **MRR for retrieval methods (all except predicate-aware):** Same source.
- [ ] **Hit@5 and Hit@10 for pre-retrieval representations:** Available in `data/results/pre_retrieval_results/` files; needed to complete Table 3.2.
- [ ] **Segmented pre-retrieval results by difficulty:** Available in `data/results/summary_by_difficulty.json`.
- [ ] **Segmented pre-retrieval results by question type:** Available in `data/results/summary_by_category.json`.
- [ ] **Segmented retrieval results (full tables):** Available in `data/results/retrieval/thesis_tables/`.

### 7.2 Missing Figure Paths

- [ ] Figure 3.1 — Full RAG pipeline overview: **does not exist; must be created.**
- [ ] Figure 3.2 — KG-to-chunk transformation: **does not exist; must be created.** Use a record from `data/intermediate/representations/papers/enriched_metadata.jsonl`.
- [ ] Figure 3.3 — Representation strategy comparison: **does not exist; must be created.**
- [ ] Figure 3.4 — NDCG grouped bar chart: **partially exists** in `data/results/thesis_figures/` as separate entity plots; a combined grouped-bar version must be generated.
- [ ] Figure 3.5 — Retrieval workflow diagram: **does not exist; must be created.**
- [ ] Figure 3.6 — Retrieval method NDCG comparison: **data exists** in `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`; figure must be generated.
- [ ] Figure 3.7 — Retrieval NDCG by entity type: **data exists**; figure must be generated.
- [ ] Figure 3.8 — Post-retrieval bridge: **does not exist; must be created.**

### 7.3 Missing Tables

- [ ] Table 3.1 — RDF predicates per entity type: **must be compiled manually** from `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`.
- [ ] Table 3.2 — Full pre-retrieval comparison: **data exists** in `data/results/thesis_tables/full_comparison.csv`; needs LaTeX/Word formatting and missing metric columns filled in.
- [ ] Table 3.3 — Best per entity: **data exists** in `data/results/thesis_tables/best_per_entity.csv`.
- [ ] Table 3.4 — Retrieval comparison: **data exists** in `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`; needs delta column.
- [ ] Tables 3.5/3.6/3.7 — Segmented retrieval tables: **data exists** in `data/results/retrieval/thesis_tables/`.

### 7.4 Unclear Methodological Decisions

- [ ] **Question type taxonomy:** The `question_type` field values are used for segmentation but never formally listed or defined anywhere in the documentation. Extract the full taxonomy from `data/questions/ml_questions_dataset.json`.
- [ ] **Difficulty assignment criteria:** How are questions labelled easy/medium/hard? Not documented. Check `data/questions/ml_questions_dataset.json` metadata or ask the question curator.
- [ ] **Combined enriched+predicate representation:** CLAUDE.md and thesis_overview.md mention that combining `enriched_metadata` and `predicate_filtered` may help hard/multi-hop questions but has not been tested as a standalone method. Clarify: is this a planned future experiment or a discarded idea?
- [ ] **Unanswerable question criterion:** 15 questions have `is_answerable = false`. What makes them unanswerable — no entity in the graph, ambiguous target, or no correct answer? Not documented.
- [ ] **ChromaDB collection structure:** Is there one Chroma collection per representation type per entity type (14 collections), or one per representation type across all entity types (14 collections)? The naming convention `papers_{representation_type}` suggests the former (entity+representation = collection), consistent with 18 collections in the DB.

### 7.5 Inconsistencies to Resolve Before Writing

| Inconsistency | Source | Resolution needed |
|---------------|--------|------------------|
| Generation metrics (SAS/ROUGE-L) appear in pre-retrieval evaluation section of CLAUDE.md | `docs/post_retrieval/CLAUDE.md` lines 94–107 | These are post-retrieval metrics; relabel them in CLAUDE.md (cosmetic only, does not affect thesis writing) |
| "question_type" vs. "category" used inconsistently | README.md uses "category"; retrieval_stage_plan.md uses "question_type" | Verify column name in `data/questions/ml_questions_dataset.json`; use whichever matches the actual data field |
| Post-retrieval LLM target: legacy code specifies LLaMA 3; current docs do not specify any LLM | Post_Retrieval_Strategy.md, archive/src/post_retrieval | Decide: is LLaMA integration planned for Chapter 4, or is the generation stage to be described as future work? |
| Pre-retrieval results path: multiple legacy paths referenced | CLAUDE.md, README.md | Canonical path is `data/results/pre_retrieval_results/`; note legacy paths exist but are not used |

### 7.6 Missing Methodology Justifications

- [ ] **Why `all-MiniLM-L6-v2` specifically?** No justification documented. Should cite: model size vs. quality trade-off, MTEB benchmark performance, reproducibility. Add a sentence in §3.4.5.
- [ ] **Why k=10?** Default top-10 is not justified anywhere. Standard RAG literature justification or ablation needed. Mark as TODO in §3.5.5.
- [ ] **Why RRF constant k=60?** Standard default in the RRF literature (Cormack et al., 2009). Should be cited explicitly.
- [ ] **Why cross-encoder `ms-marco-MiniLM-L-6-v2`?** Justified by: MS MARCO training data for passage-level relevance, size/speed trade-off. No justification in current docs.
- [ ] **Why 200,000 papers?** Gold-first subset rule is documented; the 200k cap is not justified (computational budget? coverage statistics?). Add a note in §3.4.3.

---

## 8. Final Recommended Writing Order

Follow this order to minimise revision cycles and ensure each section has the data it needs before writing.

---

### Step 1 — Collect missing metric values (pre-writing)

**What:** Read `data/results/retrieval/{method}/metrics.json` for each of the 6 methods. Extract Hit@5, Hit@10, MRR. Read `data/results/pre_retrieval_results/{entity}/{representation}/results.json` for the 5 representations with missing metrics.  
**Purpose:** Complete Tables 3.2 and 3.4 before any writing begins.  
**Files to read:** `data/results/retrieval/*/metrics.json` (6 files), `data/results/pre_retrieval_results/**` (as needed).  
**Output:** Annotated versions of Tables 3.2, 3.4, 3.5, 3.6.

---

### Step 2 — Extract question-type taxonomy (pre-writing)

**What:** Open `data/questions/ml_questions_dataset.json`. List all unique values of `question_type` and `difficulty`. Count occurrences.  
**Purpose:** Complete §3.3 and fill the segmentation dimension definitions needed for §3.4.6 and §3.5.6.  
**Output:** A small inline table listing question_type values, their counts, and which entity type they address.

---

### Step 3 — Write §3.2 (Knowledge Graph and Dataset)

**Why first:** No dependencies on experimental results. Pure description of MLSea and the question set.  
**Sources:** `data/raw/pwc_1.nt` (line count), `src/pre_retrieval/papers/raw/build_paper_records.py` (predicates), `data/questions/ml_questions_dataset.json` (question schema).  
**Insert:** Table 3.1 (RDF predicates) — compile from the extraction script.  
**Paragraph goal:** Describe the scale, structure, and annotation density of the KG; describe the question dataset schema.

---

### Step 4 — Write §3.3 (Question Set and Evaluation Design)

**Sources:** `data/questions/ml_questions_dataset.json`, `src/pre_retrieval/shared/evaluate_retrieval.py` (metric definitions).  
**Insert:** Mention 280/265 split; refer to Table 3.7 (metric definitions, to be written).  
**Paragraph goal:** Establish that evaluation is a closed-world retrieval task; define what "correct" means (target_entity_iri).

---

### Step 5 — Write §3.1 (Methodological Overview)

**Why here:** Now that you understand the components, the overview is easier to write accurately.  
**Sources:** `docs/post_retrieval/thesis_overview.md`, `docs/post_retrieval/CLAUDE.md`.  
**Insert:** Figure 3.1 (pipeline overview diagram — create this figure first).  
**Paragraph goal:** One-page bird's-eye view of the three stages; commit to terminology used throughout.

---

### Step 6 — Write §3.4.1–§3.4.3 (Motivation and Extraction)

**Sources:** `docs/post_retrieval/pre_retrieval_methodology.md`, `src/pre_retrieval/papers/raw/build_paper_records.py`.  
**Insert:** Figure 3.2 (KG-to-chunk contrast) — create this figure.  
**Paragraph goal:** Justify the entire pre-retrieval stage; explain 2-pass extraction; describe canonical record schema.

---

### Step 7 — Write §3.4.4 (Chunk Construction — all three entity types)

**Sources:** All `build_*_chunks.py` scripts; `config/pre_retrieval_config.json` (for character limits).  
**Insert:** Figure 3.3 (representation example); Table 3.1 (character limit comparison).  
**Paragraph goal:** Precise description of each of the 14 strategies; explain design rationale for enriched_metadata vs. one_hop.

---

### Step 8 — Write §3.4.5 (Embedding Generation)

**Sources:** `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py`.  
**Paragraph goal:** Justify model choice; describe ChromaDB setup; explain score = 1 − distance; mention 18 collections, 8.2 GB store.

---

### Step 9 — Write §3.4.6 (Pre-Retrieval Evaluation) and insert Table 3.2 and Table 3.3

**Sources:** `src/pre_retrieval/shared/evaluate_retrieval.py` (metric formulas), `data/results/thesis_tables/full_comparison.csv`, `best_per_entity.csv`.  
**Insert:** Table 3.2 (full comparison — now complete after Step 1), Table 3.3 (best per entity), Figure 3.4 (NDCG bar chart — generate this figure).  
**Paragraph goal:** State all metric definitions; explain evaluation loop; report main finding (entity-type-dependent best representation).

---

### Step 10 — Write §3.5.1–§3.5.2 (Retrieval Objective and Dense Baseline)

**Sources:** `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/dense_baseline.py`, `src/retrieval/config.py`.  
**Insert:** Figure 3.5 (retrieval workflow diagram — create this figure).  
**Paragraph goal:** Explain transition from pre-retrieval to retrieval; define the baseline.

---

### Step 11 — Write §3.5.3 (Hybrid Methods)

**Sources:** `src/retrieval/filtering.py`.  
**Paragraph goal:** One subsection per hybrid method; explain predicate-to-question-type mapping for predicate-aware method.

---

### Step 12 — Write §3.5.4 (RRF Methods)

**Sources:** `src/retrieval/rrf.py`, `src/retrieval/config.py` (fusion groups, k=60).  
**Paragraph goal:** Explain RRF formula with citation (Cormack et al., 2009); explain fusion group selection rationale.

---

### Step 13 — Write §3.5.5 (Top-k Candidate Generation)

**Sources:** `src/retrieval/config.py`, `src/retrieval/pipeline.py`.  
**Paragraph goal:** Justify k=10; describe output format; note handoff schema to post-retrieval.

---

### Step 14 — Write §3.5.6 (Retrieval Evaluation Metrics) and insert Tables 3.4–3.6

**Sources:** `src/retrieval/result_schema.py`, `data/results/retrieval/thesis_tables/` (all CSVs).  
**Insert:** Table 3.4 (method comparison — complete after Step 1), Table 3.5 (by difficulty), Table 3.6 (by entity type), Figure 3.6 (NDCG bar chart — generate), Figure 3.7 (by entity type — generate).  
**Paragraph goal:** Report and interpret results; explain tight clustering; motivate post-retrieval.

---

### Step 15 — Write §3.6 (Transition to Post-Retrieval)

**Sources:** `docs/post_retrieval/Post_Retrieval_Strategy.md`, `src/post_retrieval/pipeline/context_builder.py`.  
**Insert:** Figure 3.8 (post-retrieval bridge).  
**Paragraph goal:** Bridge paragraph; introduce cross-encoder, context assembly, and generation stage.

---

### Step 16 — Revision pass

- Verify all figure/table numbering is consistent throughout the chapter.
- Confirm that no metric values are stated without being verified against repository result files.
- Replace all TODO markers with confirmed values or explicit "see Section X" cross-references.
- Check terminology consistency: every occurrence of "chunk", "representation", "candidate", "entity-centric", "semantic retrieval", "symbolic filtering", "hybrid retrieval" should match the terminology established in §3.1.
- Check that NDCG is consistently described as the primary metric.
- Ensure the `pure_semantic_dense` ≡ `hybrid_type_filtering` identity result is explained (it is a system correctness confirmation, not a redundancy error).

---

*End of methodology writing execution plan.*
