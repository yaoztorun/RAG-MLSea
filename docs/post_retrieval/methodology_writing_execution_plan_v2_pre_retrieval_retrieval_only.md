# Methodology Chapter — Writing Execution Plan v2
## Pre-Retrieval and Retrieval Stages Only (RQ1 and RQ2)

**Thesis:** Retrieval-Augmented Generation over Machine Learning Knowledge Graphs  
**Institution:** KU Leuven  
**Target Chapter:** Chapter 3 — Methodology  
**Scope:** RQ1 (Pre-Retrieval Stage) and RQ2 (Retrieval Stage) only  
**Version:** v2 — improved from `methodology_writing_execution_plan.md`  
**Date:** 2026-05-05  
**All metrics and implementation details verified against repository files unless marked TODO.**

---

## Table of Contents

1. [Scope and Research Question Alignment](#1-scope-and-research-question-alignment)
2. [Chapter 3 Narrative and Argument](#2-chapter-3-narrative-and-argument)
3. [Professional Chapter 3 Outline](#3-professional-chapter-3-outline)
4. [Deep Pre-Retrieval Methodology Plan for RQ1](#4-deep-pre-retrieval-methodology-plan-for-rq1)
5. [Deep Retrieval Methodology Plan for RQ2](#5-deep-retrieval-methodology-plan-for-rq2)
6. [Knowledge Concepts Needed for Meaningful Writing](#6-knowledge-concepts-needed-for-meaningful-writing)
7. [Methodology vs Results Separation](#7-methodology-vs-results-separation)
8. [Improved Thesis-Ready Writing Blocks](#8-improved-thesis-ready-writing-blocks)
9. [Algorithm Boxes and Pseudocode](#9-algorithm-boxes-and-pseudocode)
10. [Figures and Tables Plan](#10-figures-and-tables-plan)
11. [Source-to-Section Traceability](#11-source-to-section-traceability)
12. [Citation Needs for Methodology](#12-citation-needs-for-methodology)
13. [Methodological Limitations and Validity Considerations](#13-methodological-limitations-and-validity-considerations)
14. [What Not to Write](#14-what-not-to-write)
15. [Final Checklist Before Writing Chapter 3](#15-final-checklist-before-writing-chapter-3)

---

## 1. Scope and Research Question Alignment

This document supports the methodology writing for **RQ1 (Pre-Retrieval Stage)** and **RQ2 (Retrieval Stage)** only. RQ3 (Post-Retrieval Stage) is mentioned only as a structural boundary. No post-retrieval experiments, metrics, or evaluation design are developed in this plan.

### Research Questions (Fixed — Do Not Modify)

**Main Research Question:**  
Which combination of chunking strategy, retrieval method, and RAG technique yields the best end-to-end performance for answering machine-learning questions over the MLSea knowledge graph?

**RQ1 — Pre-Retrieval Stage:**  
How do different chunking strategies for representing MLSea knowledge influence retrieval performance after embedding?

**RQ2 — Retrieval Stage:**  
Does hybrid symbolic-semantic retrieval, combining embedding-based ranking with entity-type filtering, one-hop metadata signals, predicate-aware re-ranking, and multi-representation fusion, outperform pure dense semantic retrieval?

**RQ3 — Post-Retrieval Stage (boundary only):**  
How do different post-retrieval RAG techniques influence answer accuracy, grounding, and robustness to irrelevant retrieved chunks?

---

### RQ Alignment Matrix

| Research Question | Pipeline Stage | What Is Evaluated | Methods Involved | Main Evidence Needed | Repository / Result Files | Thesis Section |
|---|---|---|---|---|---|---|
| RQ1 | Pre-Retrieval | Which textual representation of an MLSea entity best supports embedding-based retrieval | 14 representation strategies across papers (6), datasets (4), models (4) | Hit@1, Hit@5, Hit@10, MRR, NDCG per representation; best representation per entity type | `data/results/thesis_tables/full_comparison.csv`; `data/results/thesis_tables/best_per_entity.csv`; `data/results/pre_retrieval_results/` | §3.4 |
| RQ1 — RDF extraction | Pre-Retrieval | Correctness and completeness of two-pass RDF parsing | N-Triples streaming parser (papers, datasets, models) | Canonical entity records with resolved linked-entity labels | `src/pre_retrieval/papers/raw/build_paper_records.py`; `data/intermediate/raw_papers/` | §3.4.2 |
| RQ1 — entity record construction | Pre-Retrieval | Field completeness and predicate coverage of canonical records | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` | Predicate-to-field mapping tables | `src/pre_retrieval/*/raw/` | §3.4.2 |
| RQ1 — chunk representations | Pre-Retrieval | Which linearisation strategy best preserves entity semantics for embedding | 14 builder scripts, varying field selection and depth | Chunk content examples; character-limit specs | `src/pre_retrieval/*/chunking/`; `config/pre_retrieval_config.json` | §3.4.4 |
| RQ1 — embedding | Pre-Retrieval | Whether `all-MiniLM-L6-v2` encodes entity-type diversity into a shared vector space | Single shared embedder; 384-dim cosine space | Quality proxy via retrieval metrics | `src/pre_retrieval/shared/embedder.py`; `embed_and_store.py` | §3.4.5 |
| RQ1 — vector indexing | Pre-Retrieval | Whether HNSW/ChromaDB correctly stores and retrieves entity vectors | 18 Chroma collections, one per representation type | Collection integrity; 8.2 GB store with cosine metric | `src/pre_retrieval/shared/vector_store.py`; `data/intermediate/chroma/` | §3.4.5 |
| RQ1 — pre-retrieval evaluation | Pre-Retrieval | Metric validity and evaluation loop correctness | `evaluate_retrieval.py`; `aggregate_results.py` | Hit@1/5/10, MRR, NDCG per representation on 265 questions | `src/pre_retrieval/shared/evaluate_retrieval.py`; `data/results/` | §3.4.6 |
| RQ1 — representation selection | Pre-Retrieval | Which representation is selected as best per entity type for the retrieval stage | Metric-based selection: highest NDCG per entity type | Best: paper=`enriched_metadata` (0.8225), dataset=`dataset_title_only` (0.3822), model=`model_predicate_filtered` (0.8750) | `data/results/thesis_tables/best_per_entity.csv` | §3.4.7 |
| RQ2 | Retrieval | Whether symbolic-semantic hybrid methods outperform pure dense retrieval | 6 methods: 1 dense baseline + 3 hybrid + 2 RRF | Hit@1/5/10, MRR, NDCG per method across 265 questions | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`; `data/results/retrieval/summary.csv` | §3.5 |
| RQ2 — dense baseline | Retrieval | Quality of semantic ranking from best pre-retrieval representation | `pure_semantic_dense` (passthrough of pre-retrieval top-10) | NDCG 0.7337, Hit@1 0.6717 | `src/retrieval/dense_baseline.py`; `data/results/retrieval/pure_semantic_dense/` | §3.5.2 |
| RQ2 — type filtering | Retrieval | Whether entity-type consistency is preserved by pre-retrieval collections | `hybrid_type_filtering` (control) | Identical to dense baseline: NDCG 0.7337 — confirms collection purity | `src/retrieval/filtering.py`; `data/results/retrieval/hybrid_type_filtering/` | §3.5.3.1 |
| RQ2 — one-hop richness | Retrieval | Whether metadata density correlates with retrieval relevance | `hybrid_type_onehop_filtering` | NDCG 0.7351 (+0.0013) | `src/retrieval/filtering.py`; `data/results/retrieval/hybrid_type_onehop_filtering/` | §3.5.3.2 |
| RQ2 — predicate-aware | Retrieval | Whether question-type-to-predicate mapping improves precision | `hybrid_predicate_aware_filtering` | NDCG 0.7375 (+0.0038), best Hit@1 0.6868 and MRR 0.7233 | `src/retrieval/filtering.py`; `data/results/retrieval/hybrid_predicate_aware_filtering/` | §3.5.3.3 |
| RQ2 — RRF fusion | Retrieval | Whether fusing multiple representation rankings improves recall | `optional_rrf_fusion` (k=60, fuses 2–3 representations per type) | NDCG 0.7354 (+0.0017), Hit@10 0.8189 (+0.0378) | `src/retrieval/rrf.py`; `data/results/retrieval/optional_rrf_fusion/` | §3.5.4.1 |
| RQ2 — RRF + symbolic | Retrieval | Whether combining fusion breadth with predicate precision yields best NDCG | `optional_rrf_symbolic` (RRF then predicate re-rank) | Best NDCG 0.7434 (+0.0097) | `src/retrieval/rrf.py`; `data/results/retrieval/optional_rrf_symbolic/` | §3.5.4.2 |
| RQ3 | Post-Retrieval | **Out of scope for this plan; handled later.** Post-retrieval stage (re-ranking, context assembly, generation, evaluation) is not developed here. | — | — | `src/post_retrieval/` (scaffold exists, not run) | §3.6 boundary note only |

---

## 2. Chapter 3 Narrative and Argument

### The Story Chapter 3 Must Tell

The methodology chapter answers a deceptively simple question: how do we build a system that retrieves the right machine-learning entity from a knowledge graph in response to a natural-language question? Every design decision in the chapter is an answer to a *why* question, and the chapter only succeeds if these answers form a coherent, motivated sequence.

**Why MLSea cannot be used as raw RDF triples for dense retrieval.**  
The MLSea knowledge graph encodes the machine-learning literature as a collection of discrete RDF triples, each stating a single predicate-object relationship for a subject entity. While this representation is machine-interpretable and supports structured querying, it is fundamentally unsuited to dense vector retrieval. A dense retrieval system expects the corpus unit to be a coherent, semantically self-contained passage. A raw RDF triple — such as `<pwc/scientificWork/bert> <mlso:hasTaskType> <pwc/task/question-answering>` — is neither coherent nor self-contained: the subject is an opaque IRI, the predicate is a namespace token, and the object is a linked node that must itself be resolved to a human-readable label. Even if one embeds the literal string "Question Answering", the resulting vector captures only that fragment, not the entity it belongs to. The information needed to answer a question like *"What tasks does the BERT paper address?"* is distributed across dozens of triples, none of which individually constitutes a retrievable unit.

**Why entity-centric chunk construction is necessary.**  
The pre-retrieval stage exists precisely to bridge this structural gap. By aggregating all relevant triples for a single entity into a single textual string — an *entity-centric chunk* — the system creates corpus units that are semantically coherent, contextually complete, and directly comparable to natural-language questions through cosine similarity. This aggregation also resolves the two-level indirection problem inherent in RDF: linked node IRIs are resolved to their human-readable labels through a second pass over the N-Triples file, so the final chunk contains terms like "Question Answering" and "SQuAD" rather than anonymous IRIs. Without this transformation, dense retrieval over the MLSea graph would be incoherent.

**Why multiple chunking strategies are compared.**  
Entity-centric construction raises a non-trivial design question: which fields should be included in the chunk, and to what depth should linked entities be traversed? Including too few fields produces sparse representations that may lack the vocabulary needed to match question phrasing. Including too many fields risks diluting the most distinctive signal with noisy or irrelevant predicates. For example, a paper that is associated with dozens of linked datasets and methods may produce a very long chunk in which the most relevant information is buried. Different entity types compound this challenge: a paper has a rich abstract that anchors its semantic identity, whereas a dataset may have only a title and a sparse description. The thesis therefore systematically evaluates 14 representation strategies — six for papers, four for datasets, four for models — spanning a spectrum from minimal (title only) to maximal (all enriched metadata and one-hop linked entities) coverage. This comparison is the empirical backbone of RQ1.

**Why the best pre-retrieval representation per entity type becomes the input to the retrieval stage.**  
The outcome of RQ1 is not a single universal best representation, but an entity-type-specific optimal: `enriched_metadata` for papers (NDCG 0.8225), `dataset_title_only` for datasets (NDCG 0.3822), and `model_predicate_filtered` for models (NDCG 0.8750). These selections reflect a key finding: the optimal trade-off between representation specificity and noise is entity-dependent. Papers benefit from the semantic richness of their task, dataset, and method associations. Models benefit from the precision of a curated predicate whitelist that excludes noisy linked entities. Datasets, by contrast, are so sparsely annotated in the MLSea graph that adding more fields only introduces noise; the title alone carries the most concentrated retrieval signal. These entity-type-specific best representations are then used as fixed inputs for the retrieval stage, ensuring that the retrieval comparison begins from the strongest available semantic foundation.

**Why retrieval methods are compared after representation selection.**  
The retrieval stage asks a distinct question from pre-retrieval: given the best possible semantic representation, can additional signals — drawn from the knowledge graph's structural metadata — improve the ranked candidate list? This design separates two sources of retrieval quality that are often conflated in the RAG literature: the quality of the semantic representation (addressed in pre-retrieval) and the quality of the candidate generation strategy (addressed in retrieval). By fixing the representation and varying the retrieval method, the thesis isolates the contribution of symbolic and structural signals over and above pure semantic similarity.

**Why hybrid symbolic-semantic retrieval is meaningful for KG-based RAG.**  
A knowledge graph, unlike a text corpus, encodes explicit symbolic structure. Each entity has typed relationships to other entities, typed predicates, and metadata fields that carry factual semantics. A question about a paper's tasks is not just semantically related to papers that mention tasks — it is logically answered by papers that have `mlso:hasTaskType` links. A hybrid retrieval method that can exploit this structure, even partially and without a live SPARQL endpoint, can in principle achieve precision improvements that are inaccessible to pure semantic similarity. The retrieval stage evaluates several ways to incorporate such signals: filtering by expected entity type, boosting by metadata field richness (one-hop density), boosting by question-type-to-predicate alignment (predicate-aware filtering), and fusing rankings across multiple representation types (Reciprocal Rank Fusion). The empirical result — that all methods cluster within 0.0097 NDCG points of the baseline — is itself a meaningful finding: it reveals that, at this scale and with this embedding model, the representation quality is the binding constraint, and symbolic overlays provide diminishing returns.

**Why post-retrieval is not expanded in this plan.**  
Post-retrieval involves a qualitatively different research question (RQ3) that depends on experimental infrastructure — a generative language model and answer evaluation metrics — not yet integrated with the retrieval results. Including post-retrieval design in this plan would conflate planning horizons and risk premature specification of methods that may change as the retrieval findings inform downstream design choices. The boundary at §3.6 is therefore intentional: Chapter 3 concludes by delivering a ranked list of top-10 candidates per question, and Chapter 4 takes that list as its starting point.

---

## 3. Professional Chapter 3 Outline

```
3.   Methodology
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
     3.4.5  Embedding Generation and Vector Indexing
     3.4.6  Pre-Retrieval Evaluation Protocol
     3.4.7  Pre-Retrieval Representation Selection
3.5  Retrieval Phase
     3.5.1  Retrieval Objective and Design Rationale
     3.5.2  Dense Retrieval Baseline
     3.5.3  Symbolic and Metadata-Aware Hybrid Retrieval
            3.5.3.1  Type-Filtered Retrieval
            3.5.3.2  One-Hop Richness-Boosted Retrieval
            3.5.3.3  Predicate-Aware Retrieval
     3.5.4  Multi-Representation Fusion
            3.5.4.1  Reciprocal Rank Fusion
            3.5.4.2  RRF with Symbolic Filtering
     3.5.5  Top-k Candidate Generation
     3.5.6  Retrieval Evaluation Protocol
     3.5.7  Retrieval Results and Interpretation
3.6  Boundary to Post-Retrieval
```

### Section-by-Section Guide

| Section | Purpose | What to Write | RQ | Supporting Files | Figure / Table | Content Type |
|---|---|---|---|---|---|---|
| 3.1 | Orient the reader; introduce the three-stage pipeline at a high level | One-page overview of the full pipeline; commit to terminology; introduce entity-centric design principle | Both | `docs/post_retrieval/thesis_overview.md`, `CLAUDE.md` | Figure 3.1 (pipeline overview) | Methodology overview |
| 3.2 | Describe MLSea KG, N-Triples format, entity types, scale, annotation density | Describe 26.6M triple graph, three entity types, RDF predicates used, namespace heterogeneity, sparse vs. rich annotation | RQ1 | `data/raw/pwc_1.nt`; `build_paper_records.py` | Table 3.1 (predicates per entity type) | Methodology |
| 3.3 | Describe the evaluation question dataset and gold-target design | 280 questions, 265 answerable, question types, difficulty levels, `target_entity_iri` as gold standard, closed-world assumption | Both | `data/questions/ml_questions_dataset.json` | Table 3.2 (question type distribution) | Experimental setup |
| 3.4.1 | Justify why raw RDF triples cannot serve as corpus units for dense retrieval | The two problems: lexical incoherence and two-level IRI indirection | RQ1 | `docs/post_retrieval/pre_retrieval_methodology.md` | Figure 3.2 (RDF triple vs. entity chunk) | Methodology |
| 3.4.2 | Explain two-pass N-Triples streaming extraction and canonical record schema | Pass 1: entity triples; Pass 2: linked-node label resolution; canonical field schema | RQ1 | `src/pre_retrieval/papers/raw/build_paper_records.py`; `build_dataset_records.py`; `build_model_records.py`; `shared/utils.py` | Table 3.1; Figure 3.3 (RDF-to-record) | Methodology |
| 3.4.3 | Explain corpus curation logic (gold-first, 200k cap) | Gold entities always included; remaining capacity filled from full paper set; output path | RQ1 | `src/pre_retrieval/papers/raw/build_curated_subset.py` | None required | Methodology |
| 3.4.4.1 | Describe all 6 paper representation strategies | Field content, character limits, design rationale per strategy | RQ1 | `src/pre_retrieval/papers/chunking/build_*.py`; `config/pre_retrieval_config.json` | Figure 3.4 (representation examples); Table 3.3 (strategy matrix) | Methodology |
| 3.4.4.2 | Describe 4 dataset representation strategies | Same structure; highlight sparsity problem | RQ1 | `src/pre_retrieval/datasets/chunking/` | Table 3.3 (continued) | Methodology |
| 3.4.4.3 | Describe 4 model representation strategies | Same structure; highlight predicate-filtered design rationale | RQ1 | `src/pre_retrieval/models/chunking/` | Table 3.3 (continued) | Methodology |
| 3.4.5 | Explain embedding model, normalisation, ChromaDB HNSW indexing | `all-MiniLM-L6-v2`, 384-dim, L2-norm, cosine score = 1 − distance, 18 collections, 8.2 GB | RQ1 | `src/pre_retrieval/shared/embedder.py`; `embed_and_store.py`; `vector_store.py`; `config/pre_retrieval_config.json` | Figure 3.5 (embedding workflow); Table 3.4 (indexing config) | Methodology |
| 3.4.6 | Define Hit@k, MRR, NDCG; explain evaluation loop | Metric formulas; evaluation over 265 questions; segmentation by difficulty and entity type | RQ1 | `src/pre_retrieval/shared/evaluate_retrieval.py`; `aggregate_results.py` | Table 3.5 (metric definitions) | Experimental setup |
| 3.4.7 | Report best representation per entity type and justify selection | Metric-based selection: NDCG as criterion; present full comparison and entity-type-specific winners | RQ1 | `data/results/thesis_tables/full_comparison.csv`; `best_per_entity.csv` | Table 3.6 (full comparison); Table 3.7 (best per entity); Figure 3.6 (NDCG bar chart) | Results + interpretation |
| 3.5.1 | Explain the transition from pre-retrieval to retrieval stage | Fixed best representation as input; retrieval as candidate re-ranking/re-weighting problem | RQ2 | `docs/post_retrieval/retrieval_stage_plan.md`; `src/retrieval/README.md`; `src/retrieval/config.py` | Figure 3.7 (retrieval workflow) | Methodology |
| 3.5.2 | Describe `pure_semantic_dense` baseline | Passthrough of pre-retrieval top-10; establishes semantic ranking as baseline | RQ2 | `src/retrieval/dense_baseline.py` | None (part of Table 3.8) | Methodology |
| 3.5.3.1 | Describe `hybrid_type_filtering` | Type-prefix filter; control method; expected and observed no-op | RQ2 | `src/retrieval/filtering.py` | None | Methodology |
| 3.5.3.2 | Describe `hybrid_type_onehop_filtering` | Type filter + one-hop richness boost via non-empty field count | RQ2 | `src/retrieval/filtering.py` | None | Methodology |
| 3.5.3.3 | Describe `hybrid_predicate_aware_filtering` | Question-type-to-metadata-field mapping; boosted candidates sorted first | RQ2 | `src/retrieval/filtering.py` | None | Methodology |
| 3.5.4.1 | Describe `optional_rrf_fusion` | RRF formula (k=60); fusion groups per entity type; rank aggregation | RQ2 | `src/retrieval/rrf.py`; `src/retrieval/config.py` | None | Methodology |
| 3.5.4.2 | Describe `optional_rrf_symbolic` | RRF then predicate-aware re-ranking; combines breadth and precision | RQ2 | `src/retrieval/rrf.py` | None | Methodology |
| 3.5.5 | Explain top-k=10 design choice and output schema | k=10 default; candidate schema (entity_id, title, score, source); handoff format | RQ2 | `src/retrieval/config.py`; `src/retrieval/pipeline.py` | None | Methodology |
| 3.5.6 | Justify metric choice; explain segmentation | NDCG primary; MRR secondary; Hit@10 as recall upper bound; segmentation by difficulty / entity type / question type | RQ2 | `src/retrieval/result_schema.py`; `evaluate_retrieval_stage.py` | Table 3.5 (same metric definitions); Table 3.9 (retrieval method design) | Experimental setup |
| 3.5.7 | Report retrieval results and interpret | All 6 methods; NDCG 0.7337–0.7434; tight clustering; per-entity-type and per-difficulty analysis | RQ2 | `data/results/retrieval/thesis_tables/` (all CSVs) | Table 3.10 (method comparison); Table 3.11 (by difficulty); Table 3.12 (by entity type); Figure 3.8 (NDCG bar chart) | Results + interpretation |
| 3.6 | Boundary statement: describe what retrieval delivers and what post-retrieval needs | Top-10 candidate list as handoff; note that the gold entity not in top-10 cannot be recovered by any downstream re-ranker | — | `docs/post_retrieval/Post_Retrieval_Strategy.md` | None | Methodology boundary |

---

## 4. Deep Pre-Retrieval Methodology Plan for RQ1

---

### 4.1 MLSea RDF Extraction

**Purpose:** Convert the raw N-Triples RDF file into structured, query-friendly entity records that can subsequently be transformed into textual chunks.

**Input:** `data/raw/pwc_1.nt` — 6.4 GB, 26,606,202 triples, N-Triples format (one triple per line: `<subject> <predicate> <object> .`). Entity prefixes: papers at `http://w3id.org/mlsea/pwc/scientificWork/`, datasets at `http://w3id.org/mlsea/pwc/dataset/`, models at `http://w3id.org/mlsea/pwc/model/`.

**Transformation:** Two-pass streaming scan (implemented in `src/pre_retrieval/papers/raw/build_paper_records.py` and equivalents for datasets and models):
- **Pass 1:** For each line, check whether the subject IRI matches the entity prefix. If so, record the predicate-object pair under that subject. Simultaneously, track all object IRIs that are referenced by entity subjects — these are linked nodes.
- **Pass 2:** For each linked node IRI collected in Pass 1, retrieve its `rdfs:label`, `foaf:name`, `dcterms:title`, and RDF type declarations. Store these in a node label cache keyed by IRI.
- **Assembly:** For each entity subject, combine its predicate-object pairs with the node label cache to produce a canonical entity record with human-readable values.

**Output:** JSONL files at `data/intermediate/raw_{papers,datasets,models}/` — one JSON object per entity, with fully resolved field values.

**Design rationale:** Streaming line-by-line parsing is necessary because the 6.4 GB file cannot be loaded into memory. The two-pass design is necessary because linked nodes (e.g., a task IRI) may appear in the file before or after the paper that references them — a single pass would miss labels for nodes defined later in the file.

**Alternatives:** (a) Load into a triple store (GraphDB, Apache Jena) and use SPARQL to extract records — this was the original approach in `archive/`, but was abandoned because it introduced a runtime dependency on a live SPARQL endpoint. (b) Use `rdflib` for in-memory parsing — feasible for small graphs but impractical for 26.6M triples.

**Why suitable for MLSea:** The N-Triples format is a single flat stream with no graph nesting, making streaming line-by-line parsing efficient and deterministic. The two-pass design exactly matches the semantic structure of MLSea, where entities link to typed nodes that carry human-readable labels.

**Limitations:** Triples referencing entities not identifiable by the entity prefix (e.g., anonymous blank nodes) are silently skipped. Labels for some linked nodes may be missing if the node has no `rdfs:label` or equivalent predicate — in that case the IRI itself is used as a fallback label.

**Thesis-ready paragraph starter:** "The MLSea knowledge graph is distributed as a 6.4 GB N-Triples file comprising 26,606,202 triples. Due to the scale of the dataset and the two-level indirection inherent in its linked-entity structure, extraction is implemented as a two-pass streaming scan. The first pass collects all predicate-object pairs for each entity subject; the second pass resolves linked-node IRIs to human-readable labels by consulting a pre-built label cache."

**Figure/table recommendation:** Table 3.1 (RDF predicates per entity type); Figure 3.3 (schematic of two-pass extraction showing triple stream → entity record).

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py`, `src/pre_retrieval/datasets/raw/build_dataset_records.py`, `src/pre_retrieval/models/raw/build_model_records.py`, `src/pre_retrieval/shared/utils.py`

---

### 4.2 Linked-Entity Label Resolution

**Purpose:** Replace opaque linked-node IRIs with human-readable string labels so that entity chunks contain meaningful vocabulary rather than IRI strings.

**Input:** A set of linked-node IRIs collected during Pass 1 of the extraction. For example, a paper may link to a task node at `http://w3id.org/mlsea/pwc/task/question-answering`.

**Transformation:** For each linked-node IRI, search the N-Triples file for triples where that IRI is the subject and the predicate is one of `rdfs:label`, `foaf:name`, `dcterms:title`, or `schema:name`. The first non-empty literal found is used as the label. RDF type declarations (`rdf:type`) are also collected to infer the node's category (task, dataset, method, metric, implementation).

**Output:** A Python dictionary mapping each IRI to a `(label_string, category)` tuple. This cache is used during entity record assembly.

**Design rationale:** Without label resolution, a paper chunk would contain strings like `"tasks: http://w3id.org/mlsea/pwc/task/question-answering"`, which carries no lexical similarity to the phrase "Question Answering" in a question. Label resolution is what makes the chunk semantically useful for embedding.

**Why suitable for MLSea:** MLSea follows standard RDF naming conventions — nodes carry `rdfs:label` or `foaf:name` predicates for human-readable names. The label resolution step is lightweight and requires no external lookup.

**Limitations:** If a linked node has no label predicate at all (which can happen for sparsely annotated nodes), the IRI tail string is used as a fallback. This is a graceful degradation but may introduce noise in the chunk.

**Thesis-ready paragraph starter:** "Because linked entities in the MLSea graph are identified by IRIs rather than literal strings, a label resolution step is applied in the second extraction pass. For each IRI referenced by a paper, dataset, or model entity, the extraction pipeline retrieves the associated `rdfs:label` or equivalent predicate to obtain a human-readable name. This resolution is what ensures that entity chunks contain terms such as 'Question Answering' or 'SQuAD' rather than anonymous graph identifiers."

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py` (node cache assembly), `src/pre_retrieval/shared/utils.py` (IRI normalisation and category inference)

---

### 4.3 Canonical Entity Record Construction

**Purpose:** Produce a structured, normalised Python dictionary for each entity that serves as the unified input to all subsequent chunk builders.

**Input:** Per-entity predicate-object pairs from Pass 1, combined with the resolved label cache from Pass 2.

**Transformation:** Map each predicate URI to a canonical field name (e.g., `dcterms:title` → `title`; `mlso:hasTaskType` → `tasks`). Where multiple predicates map to the same field (e.g., `dcterms:title`, `rdfs:label`, `foaf:name` all produce `title`), take the first non-empty value in priority order.

**Output (paper record fields):**

| Field | Source Predicate(s) | Type |
|---|---|---|
| `paper_id` | Subject IRI (normalised) | string |
| `title` | `dcterms:title`, `rdfs:label`, `foaf:name` | string |
| `abstract` | `fabio:abstract`, `schema:description` | string |
| `publication_year` | `dcterms:issued`, `schema:datePublished` | string |
| `authors` | `dcterms:creator`, `schema:author` → resolved labels | list[str] |
| `keywords` | `dcat:keyword` | list[str] |
| `tasks` | `mlso:hasTaskType` → resolved labels | list[str] |
| `datasets` | linked DCAT Dataset nodes → resolved labels | list[str] |
| `methods` | linked method nodes → resolved labels | list[str] |
| `metrics` | linked metric nodes → resolved labels | list[str] |
| `implementations` | `mlso:hasRelatedImplementation`, `schema:codeRepository` | list[str] |
| `linked_entities` | all resolved linked nodes (predicate, label, types, category) | list[dict] |

Dataset and model records follow analogous schemas with entity-type-specific predicate sets.

**Design rationale:** A canonical record decouples extraction from chunk building. Each of the 14 chunk builder scripts reads the same canonical record and selects the fields it needs, rather than re-parsing the RDF.

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py`, `src/pre_retrieval/datasets/raw/build_dataset_records.py`, `src/pre_retrieval/models/raw/build_model_records.py`

---

### 4.4 Corpus Curation and 200k Subset Selection

**Purpose:** Reduce the full paper corpus to a computationally manageable subset while guaranteeing that every evaluation question has its gold target entity present in the retrieval corpus.

**Input:** `data/intermediate/raw_papers/papers_master.jsonl` — full set of extracted paper records. `data/questions/ml_questions_dataset.json` — 280 evaluation questions with `target_entity_iri` fields.

**Transformation (implemented in `src/pre_retrieval/papers/raw/build_curated_subset.py`):**
1. Collect all unique `target_entity_iri` values from the question dataset — these are the gold target papers.
2. Add all gold target papers to the output set unconditionally.
3. Fill remaining capacity (up to `max_papers = 200,000` from `config/pre_retrieval_config.json`) with other papers from `papers_master.jsonl` in file order.
4. Write the subset to `data/intermediate/raw_papers/papers_subset_200k.jsonl`.

**Output:** `data/intermediate/raw_papers/papers_subset_200k.jsonl` — 200,000 paper records, all gold targets guaranteed present.

**Design rationale:** The gold-first inclusion rule ensures that the pre-retrieval evaluation is a valid closed-world test: every question has its correct answer reachable in the retrieval corpus. Without this rule, some gold targets might fall outside the 200k subset, making those questions technically unanswerable in a non-informative way.

**Alternatives:** Random sampling — would not guarantee gold target inclusion. Full corpus — computationally prohibitive (the full `papers_master.jsonl` contains far more than 200k papers, and embedding all of them would require proportionally larger GPU time and storage).

**Why suitable for MLSea:** The MLSea corpus is large but the question set is finite (280 questions, ≤280 distinct gold papers). The gold-first rule adds negligible overhead and provides a principled evaluation guarantee.

**Limitations:** The 200,000 cap is a practical computational constraint, not a principled statistical choice. **TODO:** Document the actual size of `papers_master.jsonl` to quantify what fraction of the full corpus is retained. The subset introduces a closed-world evaluation: any paper not in the 200k set cannot be retrieved, which may artificially inflate Hit@10 if the 200k subset is not representative of the full distribution.

**Thesis-ready paragraph starter:** "To make the pre-retrieval experiments computationally feasible while preserving evaluation validity, a curated subset of 200,000 papers is constructed from the full MLSea corpus. The curation procedure first includes all papers that serve as gold-target entities for any evaluation question, ensuring that the closed-world retrieval evaluation is well-formed. The remaining capacity is filled with additional papers drawn from the full extracted corpus."

**Source files:** `src/pre_retrieval/papers/raw/build_curated_subset.py`, `config/pre_retrieval_config.json`

---

### 4.5 Paper Chunk Representations (6 strategies)

**Purpose:** Transform each canonical paper record into a flat UTF-8 string suitable for dense embedding, under six different field-selection and depth strategies.

**Input:** Canonical paper records from `data/intermediate/raw_papers/papers_subset_200k.jsonl`.

**Transformation (one builder script per strategy):**

| Strategy | Fields Included | Max Chars | Script |
|---|---|---|---|
| `title_only` | Title | 512 | `build_title_only_chunks.py` |
| `abstract_only` | Abstract | 1,600 | `build_abstract_only_chunks.py` |
| `title_abstract` | Title + abstract | 1,800 | `build_title_abstract_chunks.py` |
| `predicate_filtered` | Title + abstract + tasks + datasets + methods + metrics (curated subset, no authors/implementations) | 1,800 | `build_predicate_filtered_chunks.py` |
| `enriched_metadata` | Title + abstract (900 chars) + up to 5 tasks + 5 datasets + 5 methods + 5 metrics + 6 authors + 3 implementations | 2,200 | `build_enriched_paper_chunks.py` |
| `one_hop` | Title + abstract (700 chars) + up to 12 linked entities grouped by inferred category (tasks, datasets, methods, metrics, implementations) | 2,200 | `build_one_hop_paper_chunks.py` |

**Output:** JSONL files at `data/intermediate/representations/papers/{strategy_name}.jsonl`, one record per paper per strategy.

**Design rationale:**
- `title_only`: Minimum viable signal. Tests whether the paper title alone is sufficient for retrieval.
- `abstract_only`: Tests whether abstract-only retrieval (without the title anchor) is viable.
- `title_abstract`: Natural-language baseline combining the two most text-rich fields.
- `predicate_filtered`: A curated selection that includes structured ML-domain fields (tasks, datasets, methods, metrics) while excluding author lists and implementation URLs, which may introduce retrieval noise for task-related questions.
- `enriched_metadata`: Maximum field coverage. Includes all structured fields with per-field truncation to avoid embedding model saturation. The best performer for papers (NDCG 0.8225).
- `one_hop`: Graph-centric approach. Instead of field-by-field enumeration, it uses the linked-entity graph structure, grouping linked nodes by their inferred RDF type. This tests whether the relational graph structure adds retrieval value beyond explicit field annotation.

**Alternatives considered:** Concatenating all fields without truncation (produces chunks that exceed embedding model context windows and likely degrade embedding quality); sentence-level chunking (destroys entity-centric structure); random field selection (no principled basis).

**Limitations:** Character-limit truncation may cut off important content for papers with very long abstracts or many linked entities. The `one_hop` strategy depends on the quality of linked-entity label resolution — if labels are missing, the chunk degrades.

**Thesis-ready paragraph starter:** "Six representation strategies are designed for paper entities, spanning a spectrum from minimal to maximal information density. The simplest strategy, `title_only`, encodes only the paper's title (up to 512 characters) and serves as a semantic minimalist baseline. The most information-dense strategy, `enriched_metadata`, concatenates the title, a truncated abstract (900 characters), and up to five entries each for tasks, datasets, methods, and metrics, along with six authors and three implementation pointers, producing chunks of up to 2,200 characters."

**Figure/table recommendation:** Figure 3.4 (side-by-side comparison of `title_only`, `enriched_metadata`, and `one_hop` for the same paper); Table 3.3 (full strategy matrix with fields and character limits).

**Source files:** `src/pre_retrieval/papers/chunking/build_enriched_paper_chunks.py`, `build_one_hop_paper_chunks.py`, `build_predicate_filtered_chunks.py`, `build_title_only_chunks.py`, `build_abstract_only_chunks.py`, `build_title_abstract_chunks.py`; `config/pre_retrieval_config.json`

---

### 4.6 Dataset Chunk Representations (4 strategies)

**Purpose:** Construct dataset entity chunks that maximise retrieval signal given the sparse metadata typical of MLSea dataset annotations.

**Input:** Canonical dataset records from `data/intermediate/raw_datasets/`.

**Strategies:**

| Strategy | Fields Included | Notes |
|---|---|---|
| `dataset_title_only` | Title | Best performer (NDCG 0.3822); reflects sparse annotation |
| `dataset_metadata` | Title + description + tasks + related papers | Second highest NDCG (0.2657); richer but noisier |
| `dataset_predicate_filtered` | Selected predicate subset | Lowest NDCG (0.1919); selected predicates add noise |
| `dataset_enriched_metadata` | Title + description + related papers + tasks + implementations + linked entities | Complex; second in Hit@10 (0.5263) but lower NDCG (0.3243) |

**Key finding:** `dataset_title_only` is the best representation for datasets despite being the least information-dense. This is a direct consequence of MLSea's sparse dataset annotation: most dataset entities have only a title and very limited metadata. Adding more fields does not add signal — it adds noise. This finding motivates the thesis argument that representation quality is entity-type-dependent and cannot be globally optimised.

**Limitations:** The NDCG ceiling for datasets (0.3822 even for the best representation) reflects a fundamental sparsity problem in the underlying knowledge graph, not a failure of the retrieval method. Many dataset questions are intrinsically hard because the dataset entity in MLSea carries too little descriptive text to match question phrasing. **TODO:** Verify whether dataset metadata completeness could be improved by sourcing additional fields from the linked paper entities.

**Source files:** `src/pre_retrieval/datasets/chunking/`

---

### 4.7 Model Chunk Representations (4 strategies)

**Purpose:** Construct model entity chunks that maximise retrieval precision for ML model questions, particularly for questions about model task associations and linked datasets.

**Input:** Canonical model records from `data/intermediate/raw_models/`.

**Strategies:**

| Strategy | NDCG | Notes |
|---|---|---|
| `model_predicate_filtered` | **0.8750** | Best; curated predicate whitelist removes noisy linked entities |
| `model_enriched_metadata` | 0.6916 | Second; richer but includes noise from irrelevant linked nodes |
| `model_title_only` | 0.4465 | Minimal; model titles are often generic |
| `model_metadata` | 0.4733 | Minimal gain over title-only |

**Key finding:** `model_predicate_filtered` achieves the highest NDCG across all 14 representations (0.8750). ML models are most distinctively characterised by their task and dataset associations. A carefully selected predicate whitelist that includes these associations while excluding noisy linked entities (e.g., implementation nodes with generic labels) produces a maximally discriminative representation.

**Design rationale:** Model names in MLSea are often generic (e.g., "CNN", "BERT variant") and underspecified as standalone signals. The curated predicate set compensates by anchoring the representation in the model's functional role (tasks, linked datasets) rather than its name alone.

**Source files:** `src/pre_retrieval/models/chunking/`

---

### 4.8 Embedding Generation using `all-MiniLM-L6-v2`

**Purpose:** Encode all 14 sets of entity chunks into dense vector representations that can be compared to question embeddings via cosine similarity.

**Input:** JSONL chunk files from `data/intermediate/representations/{entity_type}/{strategy}.jsonl` for all 14 strategies.

**Transformation:**
1. Load the `sentence-transformers/all-MiniLM-L6-v2` model via the HuggingFace `sentence-transformers` library (`SentenceTransformerEmbedder` in `src/pre_retrieval/shared/embedder.py`).
2. For each chunk, call `model.encode(chunk_text, normalize_embeddings=True)` to produce a 384-dimensional L2-normalised vector.
3. Process in batches of 64 to manage GPU memory.

**Output:** 384-dimensional float vectors, one per entity per strategy. Stored directly in ChromaDB (next step).

**Model choice rationale:** `all-MiniLM-L6-v2` is a 6-layer distilled transformer (~23M parameters) trained on large-scale sentence-pair data and optimised for semantic similarity tasks. It achieves competitive scores on MTEB (Massive Text Embedding Benchmark) retrieval tasks while being substantially faster than full-scale models. Its open availability (no API key, permissive licence) supports reproducibility. **TODO:** Add explicit MTEB benchmark citation and score.

**Alternatives:** `all-mpnet-base-v2` (stronger but slower); `text-embedding-ada-002` (proprietary, API-dependent); `e5-large` (larger, better on some tasks). The choice of `all-MiniLM-L6-v2` reflects a deliberate efficiency preference appropriate for a thesis-scale experiment embedding 200,000+ entities across 14 strategies.

**Limitations:** The 512-token context window of the underlying BERT architecture means that chunks exceeding this limit are silently truncated at encoding time. Most entity chunks are designed to fit within this window (character limits are set conservatively), but very long abstracts may be truncated.

**Thesis-ready paragraph starter:** "All textual entity representations are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a distilled 6-layer SentenceTransformer model producing 384-dimensional dense vectors. Embeddings are L2-normalised prior to storage, ensuring that cosine similarity is equivalent to dot-product similarity. The model was selected for its favourable balance of encoding quality, computational efficiency, and open availability."

**Source files:** `src/pre_retrieval/shared/embedder.py`, `src/pre_retrieval/shared/embed_and_store.py`, `config/pre_retrieval_config.json`

---

### 4.9 ChromaDB Vector Indexing

**Purpose:** Store all entity embeddings in a persistent, queryable vector index that supports approximate nearest-neighbour retrieval by cosine similarity.

**Input:** 384-dim embeddings from the embedding step; entity metadata (entity_id, entity_type, representation_type, source_text).

**Transformation:**
1. Instantiate `ChromaVectorStore` from `src/pre_retrieval/shared/vector_store.py`.
2. For each of the 14 representation strategies, create a Chroma collection named `{entity_type}_{representation_type}` (e.g., `papers_enriched_metadata`, `datasets_dataset_title_only`, `models_model_predicate_filtered`).
3. Upsert embeddings in batches of 64 with deduplication by entity ID.
4. ChromaDB internally builds an HNSW index with `hnsw:space = cosine`.

**Output:** 18 Chroma collections (14 representation collections + auxiliary collections) stored in `data/intermediate/chroma/chroma.sqlite3` (8.2 GB on disk).

**Design rationale:** One collection per representation type ensures that retrieval experiments are perfectly controlled: querying a collection returns only entities from that representation type. Cross-collection contamination is structurally impossible. The HNSW index provides sub-linear approximate nearest-neighbour search, which is necessary for 200,000+ entity collections.

**HNSW note:** HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm that achieves high recall at logarithmic query time. ChromaDB exposes it through the `hnsw:space` configuration parameter. **TODO cite: HNSW/ANN.**

**Limitations:** ChromaDB's HNSW index does not support incremental updates efficiently — adding new entities requires rebuilding the index. The 8.2 GB store size reflects the cumulative storage of 14 independent collections; a production system would likely use a single shared collection with metadata filters.

**Source files:** `src/pre_retrieval/shared/vector_store.py`, `src/pre_retrieval/shared/embed_and_store.py`

---

### 4.10 Question Embedding

**Purpose:** Encode each evaluation question into the same 384-dimensional vector space as the entity chunks, enabling cosine-similarity-based retrieval.

**Input:** 280 questions from `data/questions/ml_questions_dataset.json` (question text field).

**Transformation:** Apply the same `SentenceTransformerEmbedder` with `all-MiniLM-L6-v2` and L2-normalisation as used for entity chunks. This is critical: if a different model or normalisation procedure were used for questions and entities, the cosine similarity comparison would be invalid.

**Output:** One 384-dim normalised vector per question.

**Design rationale:** Using the same model for both question and entity encoding is a standard practice in bi-encoder dense retrieval (SBERT-style). **TODO cite: SBERT/SentenceTransformers.** The shared embedding space ensures that semantic similarity between question and entity is well-defined.

**Limitations:** The question encoder is applied without fine-tuning on ML-domain question-answer pairs. Domain-adapted fine-tuning (e.g., using contrastive training on question-entity pairs from MLSea) could improve retrieval quality but is beyond the scope of this thesis.

**Source files:** `src/pre_retrieval/shared/retrieve.py`, `src/pre_retrieval/shared/embedder.py`

---

### 4.11 Top-10 Pre-Retrieval Candidate Generation

**Purpose:** For each question and each representation strategy, retrieve the 10 most semantically similar entity chunks from the corresponding Chroma collection.

**Input:** Question embedding (384-dim); target Chroma collection for a given representation strategy.

**Transformation:**
1. Query the collection using `collection.query(query_embeddings=[question_vector], n_results=10)`.
2. ChromaDB returns entity IDs, source texts, metadata, and cosine distances.
3. Convert distance to similarity score: `score = 1.0 - cosine_distance`.
4. Return ranked list of 10 candidates with entity_id, title, score, representation_type, and source_text.

**Output:** `top10.json` files at `data/results/pre_retrieval_results/{entity_type}/{representation}/top10.json` — one file per question per representation strategy.

**Design rationale:** Top-10 provides a recall upper bound (Hit@10) that is informative for downstream post-retrieval stages. Retrieving fewer candidates (e.g., top-5) would restrict the recall ceiling for re-ranking. Retrieving more (e.g., top-20) would add storage and computation without meaningful benefit given the single-gold-target evaluation design.

**Source files:** `src/pre_retrieval/shared/retrieve.py`

---

### 4.12 Pre-Retrieval Metric Computation

**Purpose:** Quantify the retrieval quality of each representation strategy using standard ranked-retrieval metrics computed against gold target entity IRIs.

**Input:** Top-10 candidate lists (`top10.json`) and gold target IRIs from the question dataset.

**Metrics (implemented in `src/pre_retrieval/shared/evaluate_retrieval.py`):**

| Metric | Formula (0-indexed rank r) | Interpretation |
|---|---|---|
| Hit@1 | `1 if r == 0 else 0` | Gold is the top result |
| Hit@5 | `1 if r < 5 else 0` | Gold in top 5 |
| Hit@10 | `1 if r < 10 else 0` | Gold in top 10; recall upper bound |
| MRR | `1.0 / (r + 1)` | Reciprocal of gold rank; mean across questions |
| NDCG | `1.0 / log2(r + 2)` | Discounted precision; primary metric |

**Gold target matching:** Both the gold IRI and candidate entity IRIs are URL-decoded and stripped of graph-database wrapper prefixes before comparison, to avoid false negatives from formatting differences.

**Aggregation:** Metrics are averaged over 265 answerable questions (15 unanswerable excluded). Segmentation is computed by: (a) difficulty level (`easy`, `medium`, `hard`, `unknown`); (b) entity type (`paper`, `dataset`, `model`).

**Why NDCG is primary:** NDCG weights correct answers by their rank position. A correct answer at rank 1 contributes 1.0 / log₂(2) = 1.0; at rank 5 contributes 1.0 / log₂(6) ≈ 0.39. This captures the intuition that the top-ranked candidate is most likely to be used by the downstream context builder, and that placing the gold entity at rank 1 is substantially more valuable than placing it at rank 10.

**Why Precision@k is not used:** Each question has exactly one gold entity, making Precision@k = Hit@k / k — identical up to a scaling constant and adding no information.

**Source files:** `src/pre_retrieval/shared/evaluate_retrieval.py`, `src/pre_retrieval/shared/aggregate_results.py`

---

### 4.13 Representation Selection per Entity Type

**Purpose:** Select the single best representation per entity type to serve as the fixed input to the retrieval stage, based on the pre-retrieval evaluation results.

**Input:** Aggregated NDCG values from `data/results/thesis_tables/full_comparison.csv` and `data/results/thesis_tables/best_per_entity.csv`.

**Selection criterion:** Highest NDCG score for that entity type across all its representation strategies. NDCG is used as the criterion because it is the primary metric (see §4.12) and best reflects the expected downstream utility of the ranking.

**Selection outcome:**

| Entity Type | Best Representation | NDCG | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| Paper | `enriched_metadata` | **0.8225** | 0.7753 | 0.8483 | 0.8539 | 0.8117 |
| Dataset | `dataset_title_only` | **0.3822** | 0.2807 | 0.4561 | 0.4737 | 0.3519 |
| Model | `model_predicate_filtered` | **0.8750** | 0.8000 | 0.9000 | 0.9333 | 0.8556 |

**Thesis narrative for the selection:** The selection is presented not merely as a technical handoff between pipeline stages, but as the primary finding of RQ1. The entity-type-specific variation in optimal strategy is the empirical answer to RQ1: there is no universal best representation for KG-based dense retrieval, and the optimal choice depends on the annotation density and structural characteristics of each entity type in the knowledge graph.

**Source files:** `data/results/thesis_tables/best_per_entity.csv`; `src/retrieval/config.py` (hardcodes the selected representations as `BEST_REPRESENTATIONS`)

---

## 5. Deep Retrieval Methodology Plan for RQ2

### Method Family Overview

| Family | Methods | Signal Type | Philosophy |
|---|---|---|---|
| Dense semantic baseline | `pure_semantic_dense` | Semantic only (cosine similarity) | Establishes how well the pre-retrieval representation alone drives retrieval |
| Symbolic / metadata-aware filtering | `hybrid_type_filtering`, `hybrid_type_onehop_filtering`, `hybrid_predicate_aware_filtering` | Semantic + KG structural metadata signals (offline, pre-computed) | Tests whether KG-native signals improve the semantic ranking |
| Multi-representation fusion | `optional_rrf_fusion`, `optional_rrf_symbolic` | Rank aggregation across multiple representations | Tests whether representation diversity recovers entities missed by any single representation |

**Critical implementation note:** None of the six retrieval methods uses a live SPARQL endpoint or issues any graph query at retrieval time. All symbolic signals (entity type, metadata field presence, predicate availability) are read from pre-computed `top10.json` candidate records. The pipeline is fully offline.

---

### 5.1 `pure_semantic_dense`

**Purpose:** Establish a strong semantic baseline using the best pre-retrieval representation for each entity type as the final retrieval output, without any re-ranking.

**Input:** Pre-retrieval top-10 candidate list from the best representation per entity type, from `data/results/pre_retrieval_results/{entity_type}/{best_representation}/top10.json`.

**Algorithmic logic:** Direct passthrough. The semantic ranking produced by cosine similarity in the pre-retrieval stage is returned unmodified. No re-scoring, no filtering, no re-ordering.

**Output:** Top-10 candidates ordered by descending cosine similarity score.

**Symbolic signal used:** None. Pure embedding-based ranking.

**Why included:** Every retrieval study requires a baseline. `pure_semantic_dense` represents the state of the art for a system using only semantic similarity. Any method not outperforming this fails to justify its added complexity.

**Expected strength:** High semantic precision for questions where vocabulary directly matches representation content. Robust and fast.

**Expected weakness:** Cannot exploit KG structural signals. Treats all candidates uniformly regardless of metadata richness or alignment with the question's informational intent.

**How to describe in thesis:** "The `pure_semantic_dense` method uses the cosine-similarity ranking from the pre-retrieval stage without modification. It establishes the performance ceiling achievable by semantic similarity alone, given the best-per-entity-type representation selected in §3.4.7."

**Result evidence:** NDCG 0.7337, Hit@1 0.6717, Hit@5 0.7698, Hit@10 0.7811, MRR 0.7177. Reference point for all Δ computations.

**Source files:** `src/retrieval/dense_baseline.py`; result: `data/results/retrieval/pure_semantic_dense/`

---

### 5.2 `hybrid_type_filtering`

**Purpose:** Test whether applying an explicit entity-type filter improves retrieval precision, and confirm that Chroma collections are already type-pure.

**Input:** Pre-retrieval top-10 list for the best representation per entity type.

**Algorithmic logic:**
1. Determine expected entity type from `target_entity_iri` prefix (paper / dataset / model) — offline IRI string matching, no SPARQL.
2. Filter candidates to retain only those whose `entity_type` metadata matches the expected type.
3. Fallback: if no candidates survive, return unfiltered list.
4. Return filtered candidates in original semantic order.

**Output:** Type-filtered top-10, original semantic order.

**Symbolic signal used:** Entity type derived from IRI prefix (pre-computed in candidate metadata).

**Why included:** In the pre-retrieval stage, each Chroma collection is already restricted to a single entity type. This method is a control: it confirms collection purity. Its expected result is no change from the baseline.

**Expected strength:** Structural correctness check. Confirms system integrity.

**Expected weakness:** Structural no-op in the current architecture.

**How to describe in thesis:** "`hybrid_type_filtering` applies an explicit entity-type filter to the pre-retrieval candidate list. Because Chroma collections are restricted to a single entity type by construction, this method produces results identical to the dense baseline (NDCG 0.7337), confirming the integrity of the collection architecture."

**Result evidence:** NDCG 0.7337, Hit@1 0.6717, Hit@5 0.7698, Hit@10 0.7811, MRR 0.7177. Δ = 0.0000.

**What supports/rejects usefulness:** Its no-op result is the expected and correct outcome — deviation would indicate a system bug.

**Source files:** `src/retrieval/filtering.py`; result: `data/results/retrieval/hybrid_type_filtering/`

---

### 5.3 `hybrid_type_onehop_filtering`

**Purpose:** Test whether graph-structural richness — measured as the number of populated metadata fields — correlates with retrieval relevance.

**Input:** Pre-retrieval top-10 list for the best representation per entity type.

**Algorithmic logic:**
1. Apply entity-type filter.
2. Compute one-hop richness score for each candidate:
   - Count non-empty fields among: `tasks`, `datasets`, `methods`, `metrics`, `implementations` (each scores +1).
   - Bonus +1 if `"Linked Entities"` appears in the candidate's `source_text`.
   - Total score: integer in [0, 6].
3. Re-sort by richness score descending; ties broken by original semantic rank.

**Output:** Type-filtered, richness-re-ranked top-10.

**Symbolic signal used:** One-hop graph connectivity density — count of populated KG metadata fields in the pre-computed candidate record.

**Why included:** Tests the graph-structural hypothesis: entities with richer KG connectivity are hypothesised to be more informative and potentially more relevant for multi-field questions.

**Expected strength:** May improve ranking for questions requiring multiple facts.

**Expected weakness:** Richness is a proxy signal. A well-connected but semantically off-topic entity will outrank a sparse but correct one.

**How to describe in thesis:** "`hybrid_type_onehop_filtering` augments the type-filtered semantic ranking with a graph-richness signal. Each candidate is scored by the count of populated KG metadata fields — tasks, datasets, methods, metrics, and implementations. Candidates with richer one-hop connectivity are promoted on the assumption that they are more informative for complex multi-field questions."

**Result evidence:** NDCG 0.7351 (+0.0013), Hit@1 0.6717 (unchanged), Hit@5 0.7698 (unchanged), Hit@10 0.7811 (unchanged), MRR 0.7194 (+0.0017). Marginal NDCG gain; no change in recall metrics.

**Source files:** `src/retrieval/filtering.py` (function `_onehop_richness()`); result: `data/results/retrieval/hybrid_type_onehop_filtering/`

---

### 5.4 `hybrid_predicate_aware_filtering`

**Purpose:** Exploit question-type-to-predicate alignment to promote candidates containing the metadata field most relevant to the question's informational intent.

**Input:** Pre-retrieval top-10 list; `question_type` field from the evaluation question.

**Algorithmic logic:**
1. Apply entity-type filter.
2. Map `question_type` to a target metadata field:
   - Task-related questions → boost candidates with non-empty `tasks`.
   - Implementation questions → boost candidates with non-empty `implementations` or "Linked Entities" in source text.
   - Year questions → boost candidates with non-empty `publication_year`.
   - Model/repository questions → boost candidates mentioning "Linked Entities".
3. Split into: (a) boosted (have target field) and (b) remainder. Preserve semantic order within each group.
4. Concatenate: boosted first, remainder second.
5. Fallback: if `question_type` has no mapping, return original semantic order.

**Output:** Type-filtered, predicate-boosted top-10.

**Symbolic signal used:** Question-type-to-predicate mapping; presence of specific metadata fields in candidate records (all pre-computed, offline).

**Why included:** Different question types interrogate different KG predicates. Encoding this intent as a symbolic overlay tests whether question-type awareness improves precision.

**Expected strength:** Targeted Hit@1 and MRR improvements for question types with clear predicate mappings.

**Expected weakness:** Mapping must be manually defined. Unmapped question types receive no boost.

**How to describe in thesis:** "`hybrid_predicate_aware_filtering` encodes question-type-specific retrieval intent as a symbolic re-ranking signal. A predefined mapping from question type to metadata field promotes candidates containing the relevant structured information. For example, questions categorised as `paper_to_tasks` promote candidates with non-empty task associations."

**Result evidence:** NDCG 0.7375 (+0.0038), Hit@1 0.6868 (best of all methods), Hit@5 0.7623, Hit@10 0.7811, MRR 0.7233 (best of all methods).

**What supports its usefulness:** Best Hit@1 and MRR confirm improved precision for mapped question types. Gap between Hit@1 and Hit@10 narrows slightly vs. baseline.

**Source files:** `src/retrieval/filtering.py` (function `_boost_by_predicate()`); result: `data/results/retrieval/hybrid_predicate_aware_filtering/`

---

### 5.5 `optional_rrf_fusion`

**Purpose:** Improve recall breadth by aggregating rankings from multiple representation strategies, reducing dependence on any single representation.

**Input:** Pre-retrieval top-10 lists for multiple representations per entity type (fusion groups):
- Paper: [`enriched_metadata`, `predicate_filtered`, `one_hop`]
- Dataset: [`dataset_title_only`, `dataset_enriched_metadata`]
- Model: [`model_predicate_filtered`, `model_enriched_metadata`]

**Algorithmic logic (Reciprocal Rank Fusion, k=60):**
```
RRF_score(entity) = Σ_{representation i}  1 / (60 + rank_i(entity))
```
For each entity appearing in any ranked list, sum 1/(60+rank) across all representations. Entities absent from a representation contribute 0. Sort by descending RRF score.

**Output:** RRF-fused top-10 combining entities from all fusion-group representations.

**Symbolic signal used:** Rank diversity across representation strategies — no KG-structural signal.

**Why included:** RRF is a robust rank aggregation method proven effective when multiple complementary ranked lists are available. TODO cite: RRF (Cormack et al., 2009). An entity consistently ranked highly across multiple independent representations is more likely to be genuinely relevant.

**Expected strength:** Higher Hit@5 and Hit@10 (broader recall). Confirmed: Hit@10 rises to 0.8189 (+0.0378 vs. baseline).

**Expected weakness:** May depress Hit@1 when the top-ranked entity in the single best representation is not consistently top-ranked across others. Confirmed: Hit@1 drops to 0.6491 (−0.0226 vs. baseline).

**How to describe in thesis:** "`optional_rrf_fusion` applies Reciprocal Rank Fusion (k=60) across multiple pre-retrieval rankings per entity type. This fusion strategy increases recall breadth — Hit@10 rises to 0.8189 — at the cost of precision, as evidenced by the reduction in Hit@1 to 0.6491 relative to the dense baseline. The precision-recall trade-off is a known characteristic of rank aggregation methods."

**Result evidence:** NDCG 0.7354 (+0.0017), Hit@1 0.6491, Hit@5 0.7774, Hit@10 0.8189, MRR 0.7086.

**Source files:** `src/retrieval/rrf.py`; `src/retrieval/config.py` (fusion groups, k=60); result: `data/results/retrieval/optional_rrf_fusion/`

---

### 5.6 `optional_rrf_symbolic`

**Purpose:** Combine the recall breadth of RRF fusion with the precision gains of predicate-aware filtering, testing whether the two signals are additive.

**Input:** Same fusion groups as `optional_rrf_fusion`.

**Algorithmic logic:**
1. Apply RRF fusion (identical to `optional_rrf_fusion`).
2. Apply predicate-aware boosting to the fused list: promote candidates with the question-type-specific target metadata field.

**Output:** RRF-fused, predicate-re-ranked top-10.

**Symbolic signal used:** Rank diversity across representations (RRF) + question-type-to-predicate alignment (predicate boosting).

**Why included:** If RRF and predicate boosting are complementary, their combination should achieve both recall gains and precision gains simultaneously.

**Expected strength:** Best overall NDCG. Confirmed: NDCG 0.7434 (highest of all six methods).

**Expected weakness:** Additive complexity. Predicate boosting partially overrides RRF's ranking; Hit@1 (0.6642) remains below the predicate-only method's Hit@1 (0.6868).

**How to describe in thesis:** "`optional_rrf_symbolic` combines multi-representation rank fusion with question-type-specific predicate boosting. After RRF aggregation, the fused list is re-ranked according to the predicate-aware signal. This method achieves the highest NDCG among all evaluated methods (0.7434), suggesting that semantic diversity and structural precision are partially complementary signals in the KG-RAG retrieval setting."

**Result evidence:** NDCG 0.7434 (+0.0097, best), Hit@1 0.6642, Hit@5 0.7774, Hit@10 0.8189, MRR 0.7192.
By entity type (NDCG): Paper 0.8146, Dataset 0.4645 (largest gain vs. baseline: +0.0823), Model 0.8504.

**What supports its usefulness:** The largest gains are on dataset retrieval, where semantic similarity alone is weakest. RRF across `dataset_title_only` and `dataset_enriched_metadata` combined with predicate boosting lifts dataset NDCG from 0.3822 to 0.4645. For papers and models the improvement is marginal because the baseline is already strong.

**Source files:** `src/retrieval/rrf.py`; result: `data/results/retrieval/optional_rrf_symbolic/`

---

## 6. Knowledge Concepts Needed for Meaningful Writing

---

**RDF triples**
A Resource Description Framework triple is the atomic unit of a knowledge graph: `(subject, predicate, object)`. The subject and predicate are IRIs; the object is either an IRI or a literal. A triple encodes exactly one factual relationship. Example: `<bert_paper> <dcterms:title> "BERT: Pre-training..." .`

**RDF knowledge graphs**
A knowledge graph is a directed labelled multigraph in which nodes are entities or literals and edges are typed relationships (predicates). MLSea is an RDF knowledge graph because its entire structure is expressed as RDF triples. The graph encodes the machine-learning literature: papers, datasets, models, tasks, methods, and their relationships.

**Why RDF triples are not natural retrieval units**
Each triple encodes a single fact. A natural-language question like "What tasks does BERT address?" requires knowledge of multiple linked triples — the paper entity, all its `mlso:hasTaskType` links, and the human-readable labels of those linked task nodes. No single triple captures this answer. Dense retrieval requires a coherent, context-complete unit; a triple provides neither.

**Entity-centric representation**
An entity-centric representation aggregates all relevant predicates and resolved linked-node labels for a single entity into one textual string. It is the fundamental design choice that makes dense retrieval over a KG feasible. In this thesis, entity-centric chunks are the output of the pre-retrieval stage.

**Chunk representation**
A chunk is the output of a specific builder script: a flat UTF-8 string derived from a canonical entity record by selecting specific fields, applying character limits, and concatenating into a natural-language-like format. "Chunking strategy" in this thesis refers to the field-selection policy (which predicates to include) and the graph depth (whether to follow linked-entity edges and to what depth).

**Semantic chunking in this thesis**
Unlike document chunking (splitting long texts at sentence or paragraph boundaries), the chunking in this thesis is *entity-centric semantic chunking*: each chunk corresponds to one KG entity and encodes that entity's semantic role. The "semantic" dimension refers to the selection of semantically meaningful predicates, not to natural-language sentence segmentation.

**Predicate filtering**
Predicate filtering selects a whitelist of RDF predicates to include in an entity representation. For example, `model_predicate_filtered` includes tasks and linked datasets but excludes generic metadata. The hypothesis is that including only high-signal predicates produces more discriminative embeddings than including all available predicates, because irrelevant predicates add noise to the cosine similarity computation.

**One-hop graph neighbourhood**
In a graph, the one-hop neighbourhood of a node is the set of all nodes directly connected to it by a single edge. In the `one_hop` paper representation, the one-hop neighbourhood of a paper includes all task, dataset, method, metric, and implementation nodes linked by a single predicate. This neighbourhood is linearised into the chunk text using resolved labels grouped by node category.

**Linked-entity label resolution**
IRIs in RDF graphs do not carry human-readable meaning by themselves. Label resolution is the process of traversing the graph to find the `rdfs:label`, `foaf:name`, or equivalent predicate for each linked node, substituting the IRI with its string label. Without label resolution, entity chunks would contain opaque IRIs that carry no lexical meaning for an embedding model.

**Dense retrieval**
Dense retrieval encodes both queries and corpus documents into a shared dense vector space using a neural encoder, then retrieves the most similar documents to a query by approximate nearest-neighbour search. In contrast to sparse retrieval (BM25, TF-IDF), dense retrieval captures semantic similarity beyond exact keyword overlap. TODO cite: dense retrieval.

**Embedding-based ranking**
The process of ranking retrieved candidates by their cosine similarity score to the query embedding. In this thesis, cosine similarity is computed as `score = 1.0 − cosine_distance`, where ChromaDB computes cosine distance using HNSW. Higher score = more similar = higher rank.

**Cosine similarity**
Cosine similarity between two L2-normalised vectors equals their dot product: `sim(a, b) = a · b`. It measures the angle between two vectors in high-dimensional space, independent of their magnitudes. A score of 1.0 means identical orientation (semantically identical); 0.0 means orthogonal (no semantic overlap).

**Vector store**
A vector store is a database optimised for storing dense vector embeddings and performing approximate nearest-neighbour retrieval. In this thesis, ChromaDB serves as the vector store, persisting all entity embeddings in `data/intermediate/chroma/chroma.sqlite3`.

**ChromaDB**
ChromaDB is an open-source embedding database that provides: (a) persistent storage of vectors and associated metadata; (b) HNSW-based approximate nearest-neighbour retrieval; (c) a Python-native API. It is used in this thesis in persistent mode with cosine distance. TODO cite: ChromaDB documentation.

**HNSW indexing**
Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest-neighbour algorithm that organises vectors into a multi-layer graph where short-range connections enable fast traversal to approximate nearest neighbours. HNSW achieves sub-linear query time with high recall. TODO cite: HNSW.

**Symbolic retrieval signal**
A retrieval signal derived from the structured, symbolic layer of the knowledge graph — such as entity type, predicate presence, or metadata field completeness — rather than from the continuous vector space. In this thesis, symbolic signals are all pre-computed from canonical entity records and applied at re-ranking time without graph traversal.

**Hybrid symbolic-semantic retrieval**
A retrieval approach that combines dense semantic similarity (embedding-based ranking) with symbolic signals (KG metadata). In this thesis, hybrid methods overlay symbolic signals as re-ranking criteria on top of the semantic ranking. The symbolic signals do not replace the semantic ranking; they refine it.

**Reciprocal Rank Fusion (RRF)**
RRF is a rank aggregation method that combines multiple ranked lists into a single consensus ranking. The RRF score for an entity across k ranked lists is: `Σ_i 1 / (60 + rank_i)`. The constant 60 prevents the top ranks from dominating the aggregate score. RRF is parameter-free (modulo the constant) and robust to score calibration differences across lists. TODO cite: Cormack et al., 2009.

**Hit@k**
Hit@k = 1 if the gold target entity appears within the top-k retrieved results; 0 otherwise. Hit@1 measures precision; Hit@10 measures the recall upper bound. In this thesis, Precision@k is equivalent to Hit@k / k for a single-gold-target evaluation and adds no additional information.

**MRR (Mean Reciprocal Rank)**
MRR = mean over questions of 1/(rank of gold entity). For a gold entity at rank 1: MRR contribution = 1.0. At rank 5: 0.2. At rank > 10: 0.0. MRR focuses on the single highest-ranked relevant item and is appropriate for single-answer retrieval tasks.

**NDCG (Normalised Discounted Cumulative Gain)**
NDCG = 1 / log₂(rank + 2) for a single gold entity at 0-indexed rank r. NDCG discounts the contribution of the correct answer by its rank position — rank 1 contributes 1.0, rank 5 contributes ~0.43, rank 10 contributes ~0.29. It is the primary metric in this thesis because it best reflects the downstream value of the ranking for context construction.

**Why Precision@k is less useful with one gold target**
With exactly one gold entity per question, Precision@k = Hit@k / k — a constant multiple of Hit@k that adds no new information. Precision@k is meaningful when there are multiple relevant documents per query; with a single gold target, it is redundant.

---

## 7. Methodology vs Results Separation

A common weakness in thesis methodology chapters is mixing what was done (methodology), how it was set up (experimental setup), what was found (results), and what it means (interpretation) within the same subsection. The table below specifies where each type of content belongs for every major section of Chapter 3.

| Section | Methodology Content (what & why) | Experimental Setup Content (how measured) | Results Content (what was found) | Interpretation Content (what it means) |
|---|---|---|---|---|
| §3.1 Methodological Overview | Three-stage pipeline design; entity-centric design choice | None | None | None |
| §3.2 KG and Dataset | MLSea graph structure; N-Triples format; entity types; predicate vocabularies; annotation density variation | None | None | Implication: sparsity affects representation quality (one sentence) |
| §3.3 Question Set | Question schema; gold-target design; closed-world assumption | 280 total / 265 answerable / 15 unanswerable split; difficulty distribution; question type taxonomy | None | What "answerable" means for evaluation validity |
| §3.4.1 Motivation | Why raw triples cannot be embedded; two problems (incoherence, IRI indirection) | None | None | None |
| §3.4.2 RDF Extraction | Two-pass streaming design; canonical record schema; predicate-to-field mapping | Scripts used; output paths | None | None |
| §3.4.3 Corpus Curation | Gold-first inclusion rule; 200k cap | Actual numbers: gold papers included, total in master | None | Closed-world evaluation validity note |
| §3.4.4 Chunk Construction | Field selection per strategy; character limits; design rationale per strategy | Number of chunks produced per strategy | None | None |
| §3.4.5 Embedding and Indexing | Model choice (`all-MiniLM-L6-v2`); 384-dim; L2 norm; ChromaDB/HNSW design | Collection naming convention; 18 collections; 8.2 GB store; batch size 64 | None | None |
| §3.4.6 Pre-Retrieval Evaluation Protocol | Metric definitions (Hit@k, MRR, NDCG); why NDCG is primary; why Precision@k omitted | Evaluation loop over 265 questions; segmentation dimensions (difficulty, entity type) | None — do NOT put metric values here | None |
| §3.4.7 Pre-Retrieval Representation Selection | Selection criterion (highest NDCG per entity type) | **Put full comparison table here** | Best per entity type with all metrics | Why entity-type-specific results are the core finding of RQ1 |
| §3.5.1 Retrieval Objective | Fixed best representation as input; retrieval = re-ranking problem | Pre-computed top10.json as input; offline execution | None | None |
| §3.5.2–§3.5.4 Retrieval Methods | Algorithmic logic per method; signal type; design rationale | None additional | None | None |
| §3.5.5 Top-k | Why k=10; output schema | Configuration: k=10 from `config.py` | None | Impact of top-10 ceiling on post-retrieval |
| §3.5.6 Retrieval Evaluation Protocol | Same metric definitions as §3.4.6 (can cross-reference); segmentation by entity type and question type | Same evaluation framework applied to 6 methods | None — do NOT mix method comparison into protocol section | None |
| §3.5.7 Retrieval Results and Interpretation | None (methodology content belongs in §3.5.1–3.5.5) | **Full comparison table** (all 6 methods × all metrics) | NDCG 0.7337–0.7434 range; tight clustering; per entity type; per difficulty | Tight clustering → representation quality is binding constraint; dataset retrieval improvement with RRF+symbolic; motivates post-retrieval |
| §3.6 Boundary | What the retrieval output delivers | None | None | Top-10 ceiling as hard constraint on post-retrieval; RQ3 placeholder |

**Key rules:**
- Pre-retrieval metric definitions belong in §3.4.6 — do not repeat them in §3.5.6 (cross-reference instead).
- Full pre-retrieval result table (14 strategies × 5 metrics) belongs in §3.4.7, not in §3.4.6.
- Method descriptions and algorithmic logic belong in §3.5.2–3.5.4 — do not mix result values into those sections.
- Full retrieval method comparison table belongs in §3.5.7.
- Post-retrieval answer evaluation does not appear anywhere in Chapter 3.

---

## 8. Improved Thesis-Ready Writing Blocks

> Each block below includes: (1) a polished full paragraph, (2) a shorter version for space-constrained sections, and (3) a note on what figure or table should immediately follow.

---

### Block 8.1 — Methodological Overview (§3.1)

**Full paragraph:**
This chapter describes the design and experimental evaluation of a Retrieval-Augmented Generation pipeline for machine-learning-domain question answering over the MLSea knowledge graph. The pipeline is structured into three sequential stages. In the pre-retrieval stage, entities from the MLSea RDF knowledge graph are extracted, converted into textual representations through a family of entity-centric chunking strategies, embedded into a shared dense vector space, and evaluated to identify the most effective representation per entity type. In the retrieval stage, the best-performing representation per entity type serves as the basis for comparing six candidate generation strategies — ranging from a pure semantic baseline to hybrid symbolic-semantic methods and multi-representation fusion — evaluated using Hit@1, Hit@5, Hit@10, MRR, and NDCG. In the post-retrieval stage — described in Chapter 4 — the retrieved candidates are re-ranked, assembled into a generation context, and passed to a language model to produce a final answer. The present chapter addresses the pre-retrieval and retrieval stages in full. The post-retrieval stage is introduced at the chapter's close as a boundary, motivating the transition to Chapter 4.

**Shorter version:**
This chapter describes a three-stage KG-RAG pipeline. The pre-retrieval stage converts MLSea RDF entities into textual chunks and evaluates 14 representation strategies. The retrieval stage evaluates six candidate generation strategies using the best pre-retrieval representation per entity type. The post-retrieval stage, described in Chapter 4, generates and evaluates answers from the top-10 retrieved candidates.

**Figure/table note:** Insert Figure 3.1 (full pipeline overview diagram) immediately after this paragraph.

---

### Block 8.2 — MLSea KG Description (§3.2)

**Full paragraph:**
The primary data source for this thesis is the MLSea knowledge graph, an RDF-based export of the Papers with Code repository that encodes the machine-learning literature as a set of typed entities and their inter-relationships. The graph is distributed as a single N-Triples file (`data/raw/pwc_1.nt`) comprising 26,606,202 triples and occupying 6.4 gigabytes on disk. MLSea encodes three primary entity types: scientific publications (hereafter *papers*), identified by IRIs of the form `http://w3id.org/mlsea/pwc/scientificWork/{id}`; machine learning datasets, at `http://w3id.org/mlsea/pwc/dataset/{id}`; and machine learning models, at `http://w3id.org/mlsea/pwc/model/{id}`. Entity attributes are encoded using predicates drawn from heterogeneous vocabularies, including Dublin Core Terms (`dcterms`), the Data Catalog Vocabulary (`dcat`), the Machine Learning Schema Ontology (`mlso`), the Machine Learning Schema (`mls`), FOAF, and schema.org. Papers are annotated with titles, abstracts, publication years, author lists, task types, linked datasets, methods, metrics, and implementation pointers. Datasets are annotated with titles, descriptions, and linked papers. Models are annotated with tasks, evaluation runs, and hyperparameter metadata. A critical characteristic of the graph is the high variability in annotation density across entities: while prominent papers may have dozens of linked task, dataset, and method nodes, the majority of dataset and model entities are sparsely annotated, carrying only a title and limited metadata. This sparsity has direct consequences for the design and evaluation of entity-centric chunk representations in §3.4.

**Shorter version:**
MLSea is an RDF knowledge graph derived from Papers with Code, comprising 26,606,202 N-Triples (6.4 GB). It encodes three entity types — papers, datasets, and models — using predicates from multiple vocabularies. Annotation density varies substantially: papers are rich in linked metadata, while datasets and models are sparsely annotated, a characteristic that shapes the design of pre-retrieval representations.

**Figure/table note:** Insert Table 3.1 (RDF predicates per entity type) immediately after this paragraph.

---

### Block 8.3 — Question Dataset and Gold Entity Design (§3.3)

**Full paragraph:**
The evaluation framework is built around a manually curated question dataset comprising 280 natural-language questions from the machine-learning domain, stored in `data/questions/ml_questions_dataset.json`. Each question is annotated with a unique identifier, the question text, a `question_type` label, a `target_entity_iri` pointing to the correct entity in the MLSea graph, a reference answer, a human-readable text answer, a binary `is_answerable` flag, and a difficulty label (`easy`, `medium`, `hard`, or `unknown`). Of the 280 questions, 265 are marked as answerable and are used for all metric computations; the remaining 15 unanswerable questions are excluded from averaging. The `target_entity_iri` field serves as the gold standard for all retrieval evaluation: the evaluation pipeline checks whether the gold IRI appears among the top-k retrieved candidates, after normalising both the gold and candidate IRIs to a canonical form to eliminate formatting artefacts. The question set spans **TODO: insert full question type taxonomy from JSON** question types, including paper-centric queries (e.g., `paper_to_tasks`, `paper_to_publication_year`), dataset-centric queries (e.g., `dataset_to_tasks`), and cross-entity queries. This design establishes a closed-world retrieval evaluation: the correct answer is always present in the retrieval corpus (guaranteed by the corpus curation rule in §3.4.3), and the task is to rank that answer as highly as possible.

**Shorter version:**
The evaluation uses 280 manually curated ML questions; 265 are answerable and used for metric computation. Each question specifies a `target_entity_iri` as the gold retrieval target. The evaluation is a closed-world retrieval task: the gold entity is always present in the corpus.

**Figure/table note:** Insert Table 3.2 (question type and difficulty distribution) after this paragraph.

---

### Block 8.4 — Motivation for Pre-Retrieval (§3.4.1)

**Full paragraph:**
Standard Retrieval-Augmented Generation assumes that the knowledge corpus is composed of coherent natural-language passages that can be meaningfully embedded and compared to a query through cosine similarity. The MLSea knowledge graph does not satisfy this assumption. In RDF graphs, the information associated with a single entity is distributed across many individual triples, each encoding a single predicate-object relationship. A triple such as `<pwc/scientificWork/bert> <mlso:hasTaskType> <pwc/task/question-answering>` is lexically incoherent as a standalone retrieval unit: the subject is an opaque IRI, the predicate is a namespace token, and the object is itself an IRI requiring further graph traversal to resolve to the string "Question Answering". Embedding such a triple produces a vector that captures, at best, the title fragment or the predicate name — not the entity's full semantic identity. Furthermore, the information required to answer a question such as "What natural language processing tasks does the BERT paper address?" is encoded not in a single triple but in a set of linked triples spanning the paper entity, its `mlso:hasTaskType` connections, and the label nodes of each linked task. The pre-retrieval stage addresses this structural mismatch by constructing *entity-centric chunk representations* that consolidate the distributed graph information for each entity into a single semantically coherent string suitable for dense embedding.

**Shorter version:**
Raw RDF triples cannot serve as dense retrieval units for two reasons: they are lexically incoherent (subjects and objects are IRIs, not text), and they fragment entity information across many independent facts. The pre-retrieval stage resolves both problems by aggregating each entity's distributed triple data into a single entity-centric textual chunk.

**Figure/table note:** Insert Figure 3.2 (raw RDF triple vs. assembled entity chunk contrast diagram) after this paragraph.

---

### Block 8.5 — RDF Extraction and Linked Label Resolution (§3.4.2)

**Full paragraph:**
RDF entity records are extracted from `data/raw/pwc_1.nt` using a two-pass streaming parser, implemented separately for papers (`src/pre_retrieval/papers/raw/build_paper_records.py`), datasets, and models. In the first pass, the parser scans all 26,606,202 N-Triple lines sequentially, collecting predicate-object pairs for each subject IRI that matches the entity prefix (e.g., `http://w3id.org/mlsea/pwc/scientificWork/` for papers). Simultaneously, it records all object IRIs referenced by entity subjects — these are linked nodes such as task, dataset, method, and metric entities. In the second pass, the parser collects the `rdfs:label`, `foaf:name`, and RDF type declarations for each linked-node IRI, building a label cache that maps IRIs to human-readable string labels and inferred node categories. During record assembly, entity fields are populated by mapping specific predicate URIs to canonical field names; where multiple predicates can fill the same field (e.g., `dcterms:title`, `rdfs:label`, and `foaf:name` all produce the `title` field), the first non-empty value in a priority order is used. Linked entity fields (tasks, datasets, methods, metrics) are populated using the label cache. The resulting canonical entity records are stored as JSONL files at `data/intermediate/raw_{papers,datasets,models}/`.

**Shorter version:**
A two-pass streaming parser extracts entity records from the 6.4 GB N-Triples file. The first pass collects entity predicates; the second resolves linked-node IRIs to human-readable labels via a label cache. The result is a canonical entity record per entity, stored as JSONL.

**Figure/table note:** Insert Table 3.1 (predicate-to-field mapping); refer to Figure 3.3 (extraction schematic).

---

### Block 8.6 — Corpus Curation (§3.4.3)

**Full paragraph:**
The complete MLSea corpus contains far more paper entities than can be embedded and indexed within the computational budget of a thesis-scale experiment. A curated subset of 200,000 papers is constructed by `src/pre_retrieval/papers/raw/build_curated_subset.py`. The curation procedure follows a gold-first inclusion rule: all papers that appear as `target_entity_iri` in any evaluation question are unconditionally included in the subset, regardless of their position in the source file. This guarantees that every evaluation question has its correct answer present in the retrieval corpus, making the pre-retrieval evaluation a valid closed-world test. After gold targets are included, the remaining capacity (up to the 200,000 limit) is filled with additional papers from `data/intermediate/raw_papers/papers_master.jsonl` in file order. The curated subset is written to `data/intermediate/raw_papers/papers_subset_200k.jsonl` and serves as the source for all subsequent chunk construction and embedding steps. Note that dataset and model entities are not subset-selected — the full extracted set of datasets and models is used, as their total count is substantially smaller than the paper corpus.

**Shorter version:**
A 200,000-paper subset is curated from the full MLSea corpus using a gold-first inclusion rule: all gold target papers are included unconditionally; remaining capacity is filled with other papers. This ensures closed-world evaluation validity.

---

### Block 8.7 — Entity-Centric Chunk Construction (§3.4.4, intro)

**Full paragraph:**
Entity-centric chunk construction transforms each structured canonical entity record into a flat UTF-8 string suitable for encoding by a SentenceTransformer model. The fundamental design challenge is field selection: which predicates to include, in what order, and to what depth linked entities should be traversed. Too few fields produce semantically thin representations that lack the vocabulary to match question phrasing. Too many fields dilute the most discriminative signal with irrelevant or redundant content. This trade-off motivates a systematic comparison of multiple representation strategies across all three entity types: six strategies for papers, four for datasets, and four for models. All strategies apply character-length truncation to control representation size and prevent embedding model saturation; the specific limits are configured in `config/pre_retrieval_config.json`. Each strategy is implemented as a standalone builder script in `src/pre_retrieval/{entity_type}/chunking/`, ensuring that representation construction is modular, reproducible, and independently verifiable.

**Shorter version:**
Chunk construction converts each entity record into a flat UTF-8 string by selecting fields and concatenating their values. Fourteen strategies across three entity types balance the trade-off between informational richness and representational noise.

**Figure/table note:** Insert Table 3.3 (chunk representation strategy matrix) after the entity-type subsections.

---

### Block 8.8 — Paper Representations (§3.4.4.1)

**Full paragraph:**
Six representation strategies are designed for paper entities. The `title_only` strategy (up to 512 characters) encodes solely the paper's title and serves as a semantic minimalist baseline, testing whether the most compact possible representation is sufficient for retrieval. The `abstract_only` strategy (up to 1,600 characters) deliberately omits the title to test whether the abstract carries independent retrieval signal; as the evaluation results demonstrate, this omission severely penalises performance (NDCG 0.5438 vs. 0.8225 for the best strategy), confirming that the title is an irreplaceable retrieval anchor. The `title_abstract` strategy (1,800 characters) combines both text-rich fields and serves as a natural-language baseline. The `predicate_filtered` strategy (1,800 characters) augments the title and a truncated abstract with a curated selection of structured ML metadata — tasks, linked datasets, methods, and metrics — while excluding author lists and implementation URLs that may introduce retrieval noise. The `enriched_metadata` strategy (2,200 characters) provides maximum field coverage: it concatenates the title, a truncated abstract (900 characters), up to five entries each for tasks, datasets, methods, and metrics, up to six authors, and up to three implementation pointers. This strategy achieves the best retrieval performance for papers (NDCG 0.8225, Hit@1 0.7753), attributable to the alignment between its ML-domain vocabulary (task names, dataset names, method names) and the phrasing of evaluation questions. The `one_hop` strategy (2,200 characters) takes a graph-centric approach: instead of explicit field enumeration, it linearises the one-hop linked-entity neighbourhood of the paper, grouping resolved node labels by their inferred category (tasks, datasets, methods, metrics, implementations). While it achieves the third-best NDCG (0.7642), it underperforms `enriched_metadata`, likely because the graph traversal includes some nodes with generic or uninformative labels that dilute the representation.

**Shorter version:**
Six paper representation strategies span from `title_only` (512 chars) to `enriched_metadata` (2,200 chars). The best performer is `enriched_metadata` (NDCG 0.8225), which includes the title, abstract, tasks, datasets, methods, metrics, authors, and implementations. Omitting the title (`abstract_only`) severely reduces performance (NDCG 0.5438), confirming the title's role as a semantic anchor.

**Figure/table note:** Insert Figure 3.4 (side-by-side representation examples for `title_only`, `enriched_metadata`, `one_hop`) and Table 3.3 (strategy matrix with character limits and included fields).

---

### Block 8.9 — Dataset Representations (§3.4.4.2)

**Full paragraph:**
Four representation strategies are designed for dataset entities. Unlike papers, which carry rich textual metadata including abstracts, task lists, and linked method descriptions, dataset entities in the MLSea graph are sparsely annotated: most have only a title and, in better cases, a short description and a list of associated papers or tasks. This sparsity shapes the representation comparison for datasets in a fundamental way. The `dataset_title_only` strategy (up to 512 characters) achieves the best retrieval performance for datasets (NDCG 0.3822, Hit@1 0.2807), despite being the simplest strategy. The `dataset_enriched_metadata` strategy, which includes all available metadata fields and linked entities, achieves the second-highest Hit@10 (0.5263) but a lower overall NDCG (0.3243), indicating that its broader recall comes at the cost of a less precise top-ranking. The `dataset_metadata` (NDCG 0.2657) and `dataset_predicate_filtered` (NDCG 0.1919) strategies perform progressively worse, suggesting that adding structured fields to dataset representations introduces more noise than signal when the underlying annotation is sparse. The finding that the minimalist `dataset_title_only` strategy outperforms all richer alternatives is a key result of RQ1: it demonstrates that representation quality is not monotonically increasing in information density and that annotation sparsity in the source knowledge graph sets a hard ceiling on what any representation strategy can achieve.

**Shorter version:**
Four dataset representation strategies reveal a counter-intuitive finding: the simplest strategy, `dataset_title_only`, achieves the best NDCG (0.3822). Richer representations perform worse because MLSea dataset annotations are sparse, and additional fields introduce noise rather than signal.

---

### Block 8.10 — Model Representations (§3.4.4.3)

**Full paragraph:**
Four representation strategies are designed for model entities. ML model names in the MLSea graph are often generic or under-specified as standalone identifiers (e.g., "CNN", "BERT variant"), making title-only representations insufficient for distinguishing between models. The `model_predicate_filtered` strategy addresses this by applying a curated predicate whitelist that retains task associations and linked dataset references while excluding noisy linked entities (e.g., implementation nodes with generic labels). This strategy achieves the highest NDCG across all 14 representations evaluated in this thesis (NDCG 0.8750, Hit@1 0.8000, Hit@10 0.9333), demonstrating that a carefully selected subset of a model entity's graph-linked information is more informative than either the entity name alone or its full enriched graph neighbourhood. The `model_enriched_metadata` strategy, which includes all linked entity information, achieves NDCG 0.6916 — a significant reduction compared to the predicate-filtered variant, consistent with the hypothesis that over-inclusive representations introduce noise. The `model_metadata` (NDCG 0.4733) and `model_title_only` (NDCG 0.4465) strategies confirm that minimal representations are inadequate for models, whose semantic identity depends heavily on relational context.

**Shorter version:**
Model retrieval is best served by `model_predicate_filtered` (NDCG 0.8750), the highest-performing representation across all 14 strategies. Curated predicate selection outperforms both full enrichment (NDCG 0.6916) and title-only (NDCG 0.4465), confirming that graph-aware filtering is essential for sparse-but-relational model entities.

---

### Block 8.11 — Embedding and Indexing (§3.4.5)

**Full paragraph:**
All textual entity representations are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a distilled 6-layer SentenceTransformer model (TODO cite: SBERT/SentenceTransformers) producing 384-dimensional dense vectors. Embeddings are L2-normalised prior to storage, ensuring that cosine similarity is equivalent to the vector dot product. The model was selected for its balance of encoding quality on semantic similarity benchmarks (TODO cite: MTEB), computational efficiency (~23M parameters, fast inference), and open availability without API dependency. Each textual chunk is encoded in batches of 64, with the question texts encoded using the same model and normalisation procedure to ensure comparability.

Encoded representations are stored in a ChromaDB vector store at `data/intermediate/chroma/`, using a HNSW index with cosine distance as the similarity metric. Each of the 14 representation strategies occupies a separate Chroma collection, named according to the convention `{entity_type}_{representation_type}` (e.g., `papers_enriched_metadata`). This one-collection-per-representation design ensures that retrieval experiments are structurally isolated: a query against `papers_enriched_metadata` cannot retrieve dataset or model entities, nor entities from a different paper representation. The vector store comprises 18 collections (14 representation collections plus auxiliary collections) and occupies 8.2 gigabytes on disk. At query time, the retrieval score is derived as `score = 1.0 − cosine_distance`, yielding values in [0, 1] where higher scores indicate greater semantic similarity.

**Shorter version:**
All 14 entity representations are encoded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised) and stored in ChromaDB with HNSW indexing and cosine distance. One collection per representation type ensures isolated retrieval experiments. The vector store spans 18 collections and 8.2 GB.

**Figure/table note:** Insert Figure 3.5 (embedding and indexing workflow); Table 3.4 (embedding and indexing configuration).

---

### Block 8.12 — Pre-Retrieval Evaluation Protocol (§3.4.6)

**Full paragraph:**
The pre-retrieval evaluation measures how effectively each representation strategy supports semantic retrieval of the correct gold entity. For each of the 265 answerable questions, the natural-language question text is embedded using the same `all-MiniLM-L6-v2` model and queried against the Chroma collection for the target representation. The evaluation pipeline (`src/pre_retrieval/shared/evaluate_retrieval.py`) checks whether the `target_entity_iri` from the question appears among the top-10 retrieved candidates, after normalising both IRIs to a canonical form. Five metrics are computed per question:

- **Hit@k** (k ∈ {1, 5, 10}): 1 if the gold entity appears within the top-k results, else 0.
- **MRR**: 1 / (gold rank + 1) if found in top-10, else 0.
- **NDCG**: 1 / log₂(gold rank + 2) if found in top-10, else 0.

All metrics are averaged over 265 answerable questions. NDCG is designated as the primary metric because it penalises lower-ranked correct answers in proportion to their rank, reflecting the downstream importance of top-ranked candidates for context construction. Results are additionally segmented by difficulty level (`easy`, `medium`, `hard`, `unknown`) and entity type (`paper`, `dataset`, `model`) to identify interaction effects between representation strategy and question characteristics.

**Shorter version:**
Pre-retrieval evaluation measures retrieval quality over 265 answerable questions using Hit@1/5/10, MRR, and NDCG (primary). Each question is embedded and queried against the target Chroma collection; the gold IRI is matched against top-10 candidates. Results are segmented by difficulty and entity type.

**Figure/table note:** Insert Table 3.5 (metric definitions with formulas).

---

### Block 8.13 — Pre-Retrieval Representation Selection (§3.4.7)

**Full paragraph:**
The pre-retrieval evaluation reveals that no single representation strategy achieves the best performance across all entity types. The optimal representation is entity-type-specific, determined by the annotation density and structural characteristics of each entity type in the MLSea knowledge graph. For papers, `enriched_metadata` achieves the highest NDCG (0.8225), followed by `predicate_filtered` (0.7745) and `one_hop` (0.7642). The substantial gap between `enriched_metadata` and `abstract_only` (0.5438) confirms that ML-domain structured metadata — task names, dataset names, method names — provides retrieval signal substantially beyond what the abstract text alone contributes. For datasets, `dataset_title_only` achieves the best NDCG (0.3822), with all richer representations performing worse. This counter-intuitive result reflects the sparse annotation of dataset entities in MLSea: additional fields introduce noise rather than signal. For models, `model_predicate_filtered` achieves the highest NDCG across all 14 representations (0.8750), demonstrating that curated predicate selection is more effective than either minimal or maximal representations. These three best representations — `enriched_metadata`, `dataset_title_only`, and `model_predicate_filtered` — are selected as the fixed inputs to the retrieval stage. This selection constitutes the primary empirical answer to RQ1.

**Shorter version:**
Pre-retrieval results identify entity-type-specific best representations: `enriched_metadata` for papers (NDCG 0.8225), `dataset_title_only` for datasets (NDCG 0.3822), and `model_predicate_filtered` for models (NDCG 0.8750). These selections are the primary finding of RQ1 and the fixed inputs to the retrieval stage.

**Figure/table note:** Insert Table 3.6 (full 14-strategy comparison) and Table 3.7 (best per entity type); then Figure 3.6 (NDCG bar chart across all strategies).

---

### Block 8.14 — Retrieval Objective (§3.5.1)

**Full paragraph:**
The retrieval stage investigates a question distinct from pre-retrieval: given the best possible semantic representation for each entity type, do alternative candidate generation strategies — ones that incorporate structural and symbolic signals from the knowledge graph — improve the quality of the ranked candidate list? The input to all retrieval methods is the pre-computed top-10 candidate list from the best pre-retrieval representation per entity type (`enriched_metadata` for papers, `dataset_title_only` for datasets, `model_predicate_filtered` for models), stored in `data/results/pre_retrieval_results/{entity_type}/{representation}/top10.json`. This design separates the contribution of semantic representation quality from the contribution of candidate generation strategy, enabling a controlled comparison. All six retrieval methods operate on the same pre-retrieval input; their outputs are evaluated using the same metrics (Hit@1/5/10, MRR, NDCG) on the same 265 answerable questions, with additional segmentation by entity type and difficulty. The retrieval stage is entirely offline: no live graph query, SPARQL endpoint, or external API is consulted during candidate generation.

**Shorter version:**
The retrieval stage compares six candidate generation strategies using the best pre-retrieval representation as a shared fixed input. The stage isolates the contribution of symbolic and structural signals over and above semantic similarity, in a fully offline, pre-computed evaluation.

**Figure/table note:** Insert Figure 3.7 (retrieval workflow); Table 3.9 (retrieval method design matrix).

---

### Block 8.15 — Dense Retrieval Baseline (§3.5.2)

**Full paragraph:**
The `pure_semantic_dense` method establishes the retrieval baseline by using the pre-retrieval semantic ranking directly as the retrieval output. No re-scoring, filtering, or re-ordering is applied: the top-10 candidates from the best representation per entity type are returned in their original cosine-similarity order. This method achieves NDCG 0.7337 and Hit@1 0.6717 across 265 evaluated questions, representing the quality ceiling of pure semantic similarity given the selected representations. All subsequent methods are evaluated relative to this baseline; a method that does not outperform `pure_semantic_dense` on the primary metric (NDCG) fails to justify its additional complexity.

**Shorter version:**
`pure_semantic_dense` passes through the pre-retrieval semantic ranking without modification. It achieves NDCG 0.7337, Hit@1 0.6717 — the reference point against which all hybrid methods are evaluated.

---

### Block 8.16 — Type-Filtered Retrieval (§3.5.3.1)

**Full paragraph:**
The `hybrid_type_filtering` method applies an explicit entity-type filter to the pre-retrieval candidate list: candidates whose entity type (derived from their IRI prefix) does not match the expected type of the question's gold entity are removed. This method is designed as a control experiment. In the pre-retrieval stage, each Chroma collection already restricts retrieval to a single entity type by construction; the type-filtering method therefore tests whether the collection architecture is functioning correctly. The empirical result confirms this: `hybrid_type_filtering` produces results identical to the dense baseline (NDCG 0.7337, Hit@1 0.6717), with a Δ of exactly 0.0000. This identity is an expected and meaningful result — it validates the collection purity assumption that underlies the entire retrieval stage design.

**Shorter version:**
`hybrid_type_filtering` applies an entity-type filter that is a no-op in the current architecture, because Chroma collections are already type-pure. Identical results to the baseline (Δ NDCG = 0.0000) confirm collection integrity.

---

### Block 8.17 — One-Hop Richness Retrieval (§3.5.3.2)

**Full paragraph:**
The `hybrid_type_onehop_filtering` method re-ranks type-filtered candidates by a graph-richness score computed from the number of populated KG metadata fields in each candidate's pre-computed record. A candidate scores one point for each non-empty field among tasks, datasets, methods, metrics, and implementations, plus a bonus point if the representation contains linked entity text. The richness signal operationalises a graph-structural hypothesis: entities with denser one-hop connectivity in the KG are hypothesised to be more informative — and therefore more likely to be relevant — for queries that require structured ML knowledge. Empirically, `hybrid_type_onehop_filtering` achieves a marginal NDCG gain of +0.0013 over the baseline (0.7351 vs. 0.7337), with no change in Hit@1 or Hit@10. The improvement in MRR (+0.0017) suggests a slight improvement in the rank of correct answers without recovering new correct answers from outside the original top-10.

**Shorter version:**
`hybrid_type_onehop_filtering` re-ranks by KG metadata density (count of populated fields). It achieves a marginal NDCG gain (+0.0013) with no change in recall metrics, suggesting that metadata richness is a weak but directionally correct relevance proxy.

---

### Block 8.18 — Predicate-Aware Retrieval (§3.5.3.3)

**Full paragraph:**
The `hybrid_predicate_aware_filtering` method encodes question-type-specific retrieval intent as a symbolic re-ranking signal. A predefined mapping relates each `question_type` label to a target metadata field: task-related questions promote candidates with non-empty task associations; implementation questions promote candidates with non-empty implementation links; year questions promote candidates with a non-empty publication year. Within the re-ranked list, boosted candidates (those possessing the target field) are placed ahead of non-boosted candidates, preserving the original semantic order within each group. This method achieves the best Hit@1 (0.6868) and MRR (0.7233) of all six evaluated methods, indicating that question-type-to-predicate alignment is an effective precision signal: it places the correct entity at rank 1 more frequently than any other method. The NDCG improvement (+0.0038 to 0.7375) is modest but consistent, and the Hit@10 remains unchanged (0.7811), confirming that the method improves precision without broadening recall.

**Shorter version:**
`hybrid_predicate_aware_filtering` boosts candidates matching a question-type-specific metadata field. It achieves the best Hit@1 (0.6868) and MRR (0.7233), improving precision for question types with defined predicate mappings. NDCG improves to 0.7375 (+0.0038); recall is unchanged.

---

### Block 8.19 — RRF Fusion (§3.5.4.1)

**Full paragraph:**
The `optional_rrf_fusion` method applies Reciprocal Rank Fusion (RRF) to aggregate candidate rankings from multiple pre-retrieval representation strategies for the same entity type. For papers, three representations are fused (`enriched_metadata`, `predicate_filtered`, `one_hop`); for datasets, two (`dataset_title_only`, `dataset_enriched_metadata`); for models, two (`model_predicate_filtered`, `model_enriched_metadata`). The RRF score for each candidate entity is computed as the sum of 1/(60 + rank_i) across all representations in which it appears, where k=60 is the standard smoothing constant. Candidates are then re-ranked by descending RRF score. The RRF method is motivated by the intuition that an entity consistently ranked highly across multiple independent representation strategies is more likely to be genuinely relevant than one ranked highly by only a single strategy. The results confirm a characteristic RRF trade-off: Hit@10 improves to 0.8189 (+0.0378 vs. baseline), reflecting broader recall, while Hit@1 drops to 0.6491 (−0.0226), reflecting reduced precision at the top of the ranked list. NDCG improves marginally to 0.7354 (+0.0017).

**Shorter version:**
`optional_rrf_fusion` aggregates rankings from 2–3 representations per entity type using RRF (k=60). It improves recall (Hit@10: 0.8189, +0.0378) at the cost of precision (Hit@1: 0.6491, −0.0226). NDCG improves marginally to 0.7354.

**Figure/table note:** Insert Table 3.10 (full retrieval method comparison) in §3.5.7.

---

### Block 8.20 — RRF plus Symbolic Filtering (§3.5.4.2)

**Full paragraph:**
The `optional_rrf_symbolic` method combines the rank diversity of RRF fusion with the precision signal of predicate-aware filtering. After applying RRF fusion (identical to `optional_rrf_fusion`), the fused candidate list is re-ranked using the same question-type-to-predicate boosting logic as `hybrid_predicate_aware_filtering`. This sequential composition tests whether the two signals are complementary — whether the recall breadth of RRF and the precision focus of predicate boosting are additive in their effect on the primary metric. `optional_rrf_symbolic` achieves the highest NDCG of all six evaluated methods (0.7434, +0.0097 vs. baseline), supporting the complementarity hypothesis. The most notable entity-type-specific result is for datasets: `optional_rrf_symbolic` lifts dataset retrieval NDCG from 0.3822 (baseline) to 0.4645, a gain of +0.0823, suggesting that the fusion of `dataset_title_only` and `dataset_enriched_metadata` recovers some of the recall lost due to dataset metadata sparsity.

**Shorter version:**
`optional_rrf_symbolic` applies RRF fusion followed by predicate-aware re-ranking. It achieves the highest NDCG overall (0.7434, +0.0097) and the largest gain for dataset retrieval (0.4645 vs. 0.3822 baseline), suggesting partial complementarity between diversity and precision signals.

---

### Block 8.21 — Retrieval Evaluation Protocol (§3.5.6)

**Full paragraph:**
The retrieval stage evaluation applies the same metric framework as the pre-retrieval stage — Hit@1, Hit@5, Hit@10, MRR, and NDCG — to 265 answerable questions across all six retrieval methods. NDCG remains the primary metric. In addition to overall metrics, retrieval results are segmented by: (a) entity type (`paper`, `dataset`, `model`) to reveal which method families benefit specific entity categories; (b) difficulty level (`easy`, `medium`, `hard`, `unknown`) to reveal whether symbolic signals are more effective for harder questions; and (c) question type to identify which question categories benefit most from structured metadata boosting. The evaluation is implemented in `src/retrieval/evaluate_retrieval_stage.py` and produces per-method `results.json` (per-question candidates and metrics) and `metrics.json` (aggregated metrics) files under `data/results/retrieval/{method_name}/`.

**Shorter version:**
Six retrieval methods are evaluated on 265 questions using Hit@1/5/10, MRR, and NDCG, with segmentation by entity type, difficulty, and question type. Output is stored per-method in `data/results/retrieval/`.

---

### Block 8.22 — Retrieval Results Interpretation (§3.5.7)

**Full paragraph:**
The six retrieval methods produce NDCG values in the narrow range 0.7337–0.7434, a spread of 0.0097 NDCG points across 265 evaluated questions. The tight clustering of results is the most significant finding of the retrieval stage: it reveals that, given the strong semantic representations selected in §3.4.7, the symbolic and structural signals available in the pre-computed candidate metadata provide only marginal additional retrieval value. The binding constraint on retrieval quality is not the candidate generation strategy but the quality of the semantic embedding.

Among the six methods, `pure_semantic_dense` and `hybrid_type_filtering` produce identical results (Δ = 0.0000), confirming collection purity. `hybrid_predicate_aware_filtering` achieves the best precision metrics (Hit@1 0.6868, MRR 0.7233), demonstrating that question-type-to-predicate alignment is a useful but narrow signal. `optional_rrf_symbolic` achieves the best overall NDCG (0.7434) by combining the recall breadth of multi-representation fusion with the precision of predicate boosting; its largest benefit is for dataset retrieval (+0.0823 NDCG), the entity type for which semantic similarity alone is weakest.

The difficulty breakdown reveals that all methods perform comparably for easy questions (NDCG ≥ 0.98) and that the performance gap among methods widens slightly for medium questions (NDCG 0.77–0.80). Hard questions remain challenging across all methods (NDCG 0.47–0.50), suggesting that hard questions require semantic or structural signals not captured by the current representation strategies or retrieval methods. This finding motivates the post-retrieval re-ranking stage, which applies a cross-encoder to re-evaluate candidate relevance at higher computational cost.

**Shorter version:**
All six retrieval methods cluster within 0.0097 NDCG points of the baseline (0.7337–0.7434), indicating that semantic representation quality is the binding constraint. `optional_rrf_symbolic` achieves the best NDCG (0.7434); `hybrid_predicate_aware_filtering` achieves the best precision (Hit@1 0.6868). Hard questions remain challenging across all methods, motivating post-retrieval re-ranking.

**Figure/table note:** Insert Table 3.10 (full method comparison), Table 3.11 (by difficulty), Table 3.12 (by entity type), Figure 3.8 (NDCG comparison bar chart).

---

### Block 8.23 — Boundary to Post-Retrieval (§3.6)

**Full paragraph:**
The retrieval stage delivers, for each evaluation question, a ranked list of up to ten candidate entities, each characterised by its entity IRI, its textual representation, and a retrieval score. This ranked list constitutes the sole input to the post-retrieval stage, which is described in Chapter 4. An important implication of this architecture is that the retrieval stage imposes a hard upper bound on downstream performance: if the gold target entity does not appear within the top-10 candidates, no post-retrieval re-ranking method can recover it. The Hit@10 values reported in §3.5.7 — ranging from 0.7811 to 0.8189 depending on the retrieval method — represent this upper bound: approximately 18–22% of evaluation questions have gold entities that are absent from the top-10 and therefore irrecoverable at the post-retrieval stage, regardless of re-ranking quality. The post-retrieval stage (RQ3) is outside the scope of this chapter. Its design, implementation, and evaluation are addressed in Chapter 4.

**Shorter version:**
The retrieval stage produces a top-10 candidate list per question; this is the hard input boundary for post-retrieval. Entities absent from the top-10 cannot be recovered by any downstream re-ranker. Hit@10 (0.78–0.82) defines the recall ceiling. RQ3 and post-retrieval evaluation are addressed in Chapter 4.

**Figure/table note:** No figure required; this is a brief bridge section.

---

## 9. Algorithm Boxes and Pseudocode

> These pseudocode blocks are designed for direct conversion to LaTeX `algorithm` / `algorithmic` environments.

---

### Algorithm 1 — RDF Extraction and Linked-Entity Label Resolution

```
Input:  N-Triples file F, entity prefix P
Output: entity_records dict, label_cache dict

entity_records ← {}
linked_node_iris ← set()

// Pass 1: collect entity triples and linked nodes
for each line L in F:
    (subject, predicate, object) ← parse_triple(L)
    if subject starts with P:
        entity_records[subject][predicate] ← object
        if object is IRI:
            linked_node_iris.add(object)

// Pass 2: resolve linked-node labels
label_cache ← {}
for each line L in F:
    (subject, predicate, object) ← parse_triple(L)
    if subject in linked_node_iris:
        if predicate in {rdfs:label, foaf:name, dcterms:title}:
            if subject not in label_cache:
                label_cache[subject] ← {label: object, types: []}
        if predicate == rdf:type:
            label_cache[subject].types.append(object)

return entity_records, label_cache
```

Source: `src/pre_retrieval/papers/raw/build_paper_records.py`

---

### Algorithm 2 — Canonical Entity Record Construction

```
Input:  entity_records dict, label_cache dict, predicate_field_map
Output: canonical_records list

canonical_records ← []
for each (entity_id, triples) in entity_records:
    record ← {paper_id: normalise(entity_id)}
    for each (field_name, candidate_predicates) in predicate_field_map:
        for predicate in candidate_predicates:
            if triples[predicate] is non-empty:
                record[field_name] ← triples[predicate]
                break
    // Resolve linked-entity lists
    for each linked_predicate in [mlso:hasTaskType, dcat:Dataset, ...]:
        linked_iris ← triples[linked_predicate]
        record[linked_field] ← [label_cache[iri].label for iri in linked_iris
                                 if iri in label_cache]
    canonical_records.append(record)

return canonical_records
```

Source: `src/pre_retrieval/papers/raw/build_paper_records.py`, `shared/utils.py`

---

### Algorithm 3 — Chunk Construction

```
Input:  canonical_records list, representation_config (field_limits, char_limits)
Output: chunks list

chunks ← []
for each record in canonical_records:
    text ← ""
    for each (field_name, max_items, item_max_chars, separator) in representation_config:
        value ← record.get(field_name, "")
        if value is list:
            items ← [truncate(v, item_max_chars) for v in value[:max_items]]
            field_text ← field_name + ": " + separator.join(items)
        else:
            field_text ← truncate(value, representation_config.field_char_limit)
        text ← text + field_text + "\n"
    text ← text[:representation_config.total_char_limit]
    chunks.append({entity_id: record.paper_id,
                   representation_type: config.name,
                   chunk_text: text,
                   char_count: len(text)})

return chunks
```

Source: `src/pre_retrieval/papers/chunking/build_enriched_paper_chunks.py` (representative)

---

### Algorithm 4 — Embedding and ChromaDB Indexing

```
Input:  chunks list, embedder (all-MiniLM-L6-v2), collection_name, vector_store
Output: populated ChromaDB collection

embedder ← SentenceTransformerEmbedder("all-MiniLM-L6-v2")
collection ← vector_store.get_or_create_collection(
                  name=collection_name,
                  metadata={"hnsw:space": "cosine"})

for batch in chunks split into batches of size 64:
    texts ← [chunk.chunk_text for chunk in batch]
    embeddings ← embedder.encode(texts, normalize_embeddings=True)
    ids ← [chunk.entity_id for chunk in batch]
    metadatas ← [chunk.metadata for chunk in batch]
    collection.upsert(ids=ids,
                      embeddings=embeddings,
                      documents=texts,
                      metadatas=metadatas)
```

Source: `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py`

---

### Algorithm 5 — Pre-Retrieval Evaluation

```
Input:  questions list, collection, embedder, top_k=10
Output: results list (per-question metrics)

results ← []
for each question Q in questions:
    if not Q.is_answerable: continue
    q_embedding ← embedder.encode(Q.question, normalize_embeddings=True)
    candidates ← collection.query(q_embedding, n_results=top_k)
    gold_iri ← normalise(Q.target_entity_iri)
    gold_rank ← None
    for rank, candidate in enumerate(candidates):
        if normalise(candidate.entity_id) == gold_iri:
            gold_rank ← rank
            break
    results.append({
        question_id: Q.id,
        found_gold: gold_rank is not None,
        gold_rank: gold_rank,
        hit_at_1: 1 if gold_rank == 0 else 0,
        hit_at_5: 1 if gold_rank < 5 else 0,
        hit_at_10: 1 if gold_rank < 10 else 0,
        mrr: 1.0 / (gold_rank + 1) if gold_rank is not None else 0.0,
        ndcg: 1.0 / log2(gold_rank + 2) if gold_rank is not None else 0.0
    })
return results
```

Source: `src/pre_retrieval/shared/evaluate_retrieval.py`, `src/retrieval/result_schema.py`

---

### Algorithm 6 — Dense Retrieval Baseline (`pure_semantic_dense`)

```
Input:  pre_retrieval_entry (question + top10 candidates from best representation)
Output: ranked_candidates (top-10, unmodified)

// The dense baseline is a passthrough
ranked_candidates ← pre_retrieval_entry.raw_candidates  // already sorted by cosine score
return ranked_candidates[:10]
```

Source: `src/retrieval/dense_baseline.py`

---

### Algorithm 7 — Predicate-Aware Filtering (`hybrid_predicate_aware_filtering`)

```
Input:  candidates list, question_type string
Output: re-ranked candidates list

// Map question_type to target metadata field
field_map ← {
    "paper_to_tasks": "tasks",
    "task_to_dataset": "tasks",
    "paper_to_implementation": "implementations",
    "repository_to_model": "linked_entities",
    "paper_to_publication_year": "publication_year",
    ...
}
target_field ← field_map.get(question_type, None)

if target_field is None:
    return candidates  // fallback: original semantic order

boosted ← []
remainder ← []
for candidate in candidates:
    if candidate.metadata[target_field] is non-empty:
        boosted.append(candidate)
    else:
        remainder.append(candidate)

return boosted + remainder  // boosted candidates ranked first
```

Source: `src/retrieval/filtering.py` (function `_boost_by_predicate()`)

---

### Algorithm 8 — Reciprocal Rank Fusion (`optional_rrf_fusion`)

```
Input:  ranked_lists dict[representation_name → ranked candidates], k=60
Output: fused_candidates list

rrf_scores ← defaultdict(float)
for representation_name, ranked_list in ranked_lists:
    for rank, candidate in enumerate(ranked_list):
        rrf_scores[candidate.entity_id] += 1.0 / (k + rank)

// Collect all candidates (union across representations)
all_candidates ← {}
for representation_name, ranked_list in ranked_lists:
    for candidate in ranked_list:
        if candidate.entity_id not in all_candidates:
            all_candidates[candidate.entity_id] ← candidate

// Sort by RRF score descending
fused_candidates ← sorted(all_candidates.values(),
                           key=lambda c: rrf_scores[c.entity_id],
                           reverse=True)
return fused_candidates[:10]
```

Source: `src/retrieval/rrf.py`

---

### Algorithm 9 — Retrieval Stage Evaluation

```
Input:  questions list, retrieval_method function, pre_retrieval_top10 dict
Output: per_method_results list, aggregated_metrics dict

per_method_results ← []
for each question Q in questions:
    if not Q.is_answerable: continue
    entry ← load_pre_retrieval_entry(Q, pre_retrieval_top10)
    candidates ← retrieval_method(entry)
    gold_rank ← find_gold_rank(candidates, Q.target_entity_iri)
    per_method_results.append(compute_metrics(Q, gold_rank))

// Aggregate overall
aggregated_metrics["overall"] ← mean_metrics(per_method_results)
// Segment by difficulty
for difficulty in ["easy", "medium", "hard", "unknown"]:
    subset ← [r for r in per_method_results if r.difficulty == difficulty]
    aggregated_metrics[difficulty] ← mean_metrics(subset)
// Segment by entity type
for entity_type in ["paper", "dataset", "model"]:
    subset ← [r for r in per_method_results if r.entity_type == entity_type]
    aggregated_metrics[entity_type] ← mean_metrics(subset)
// Segment by question type
for qt in unique_question_types:
    subset ← [r for r in per_method_results if r.question_type == qt]
    aggregated_metrics[qt] ← mean_metrics(subset)

return per_method_results, aggregated_metrics
```

Source: `src/retrieval/evaluate_retrieval_stage.py`, `src/retrieval/aggregate_retrieval_stage.py`

---

## 10. Figures and Tables Plan

---

### Figures

| # | Title | Thesis Purpose | RQ | Insert After | Content | Type | Data Source | Exists? | LaTeX Label |
|---|---|---|---|---|---|---|---|---|---|
| Fig 3.1 | Full Pipeline Overview | Introduces three-stage KG-RAG architecture at a glance | Both | §3.1 para 1 | KG Input → Pre-Retrieval (extract/chunk/embed/evaluate) → Retrieval (6 methods) → Post-Retrieval (boundary) | Methodology | Drawn (draw.io / TikZ) | No — must create | `fig:pipeline_overview` |
| Fig 3.2 | RDF Triple vs. Entity Chunk | Motivates the entity-centric design choice | RQ1 | §3.4.1 | Left: 3–4 raw triples for BERT. Right: `enriched_metadata` chunk for BERT. Labels highlight the IRI-resolution problem. | Methodology | `data/intermediate/representations/papers/enriched_metadata.jsonl` (one example) | No — must create | `fig:rdf_to_chunk` |
| Fig 3.3 | Two-Pass RDF Extraction | Explains extraction architecture | RQ1 | §3.4.2 | Flow: N-Triples stream → Pass 1 (entity triples) → linked-node IRI set → Pass 2 (label cache) → canonical record | Methodology | Drawn | No — must create | `fig:extraction_pipeline` |
| Fig 3.4 | Example Chunk Representations | Shows what each representation looks like for the same paper | RQ1 | §3.4.4.1 | Three text boxes: same paper in `title_only`, `enriched_metadata`, `one_hop` — side by side | Methodology | `data/intermediate/representations/papers/` (one paper, three strategies) | No — must create | `fig:chunk_examples` |
| Fig 3.5 | Embedding and Vector Indexing Workflow | Explains the embedding and storage pipeline | RQ1 | §3.4.5 | Chunk texts → SentenceTransformer → 384-dim vectors → ChromaDB HNSW per collection | Methodology | Drawn | No — must create | `fig:embedding_workflow` |
| Fig 3.6 | Pre-Retrieval Evaluation Workflow | Shows the evaluation loop for each representation strategy | RQ1 | §3.4.6 | Question → embed → query Chroma collection → top-10 → compare gold IRI → compute metrics | Experimental setup | Drawn | No — must create | `fig:preretrieval_eval_workflow` |
| Fig 3.7 | Retrieval Method Workflow | Shows the retrieval stage flow for all six methods | RQ2 | §3.5.1 | Pre-retrieval top-10 → apply method (passthrough / filter / RRF) → re-ranked top-10 → metrics | Methodology | Drawn | No — must create | `fig:retrieval_workflow` |
| Fig 3.8 | NDCG Comparison — All Retrieval Methods | Visualises the tight performance clustering and ranking among methods | RQ2 | §3.5.7 | Horizontal bar chart: 6 methods on y-axis, NDCG on x-axis (0.72–0.75 range); baseline marked | Results | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` | Data exists; figure must be generated | `fig:retrieval_ndcg_comparison` |

**Recommended captions (LaTeX-ready):**

- Fig 3.1: "Overview of the three-stage KG-RAG pipeline. The pre-retrieval stage converts MLSea RDF entities into textual representations and evaluates embedding-based retrieval quality. The retrieval stage compares candidate generation strategies. The post-retrieval stage (Chapter~\ref{ch:post_retrieval}) generates and evaluates answers."
- Fig 3.2: "Contrast between a raw RDF representation (left) and an entity-centric chunk (right) for the same scientific publication. Raw triples contain opaque IRIs; the entity-centric chunk contains human-readable text suitable for dense embedding."
- Fig 3.8: "NDCG scores for all six retrieval methods evaluated on 265 answerable questions. The dashed line marks the `pure\_semantic\_dense` baseline (0.7337). The narrow performance band (0.0097 NDCG points) indicates that semantic representation quality is the binding constraint."

---

### Tables

| # | Title | Thesis Purpose | RQ | Insert After | Content | Type | Data Source | Exists? | LaTeX Label |
|---|---|---|---|---|---|---|---|---|---|
| Tab 3.1 | RDF Predicates Per Entity Type | Documents which KG predicates are used and how they map to canonical fields | RQ1 | §3.2 or §3.4.2 | Columns: Entity type \| Predicate URI \| Canonical field \| Example value | Methodology | `src/pre_retrieval/*/raw/build_*_records.py` | No — compile manually | `tab:rdf_predicates` |
| Tab 3.2 | Question Type and Difficulty Distribution | Describes the evaluation question set | Both | §3.3 | Columns: Question type \| Count \| Entity type addressed \| Difficulty breakdown | Experimental setup | `data/questions/ml_questions_dataset.json` | **TODO: extract** | `tab:question_distribution` |
| Tab 3.3 | Chunk Representation Strategy Matrix | Catalogues all 14 strategies with design parameters | RQ1 | §3.4.4 | Columns: Entity type \| Strategy name \| Fields included \| Max chars \| Script | Methodology | `src/pre_retrieval/*/chunking/`; `config/pre_retrieval_config.json` | No — compile manually | `tab:representation_matrix` |
| Tab 3.4 | Embedding and Indexing Configuration | Documents all hyperparameters of the embedding/indexing step | RQ1 | §3.4.5 | Rows: Model, dimensionality, normalisation, similarity metric, index type, collections, storage size, batch size | Experimental setup | `config/pre_retrieval_config.json`, `src/pre_retrieval/shared/` | No — compile manually | `tab:embedding_config` |
| Tab 3.5 | Retrieval Metric Definitions | Formally defines all evaluation metrics | Both | §3.4.6 | Columns: Metric \| Formula (0-indexed rank r) \| Range \| Primary use | Experimental setup | `src/pre_retrieval/shared/evaluate_retrieval.py` | No — write manually | `tab:metric_definitions` |
| Tab 3.6 | Full Pre-Retrieval Representation Comparison | Reports retrieval quality for all 14 strategies | RQ1 | §3.4.7 | Columns: Entity type \| Strategy \| Hit@1 \| Hit@5 \| Hit@10 \| MRR \| NDCG; best per entity type bolded | Results | `data/results/thesis_tables/full_comparison.csv` | **Data exists** | `tab:preretrieval_full` |
| Tab 3.7 | Best Representation Per Entity Type | Summarises RQ1 answer | RQ1 | §3.4.7 (after Tab 3.6) | Columns: Entity type \| Best strategy \| Hit@1 \| Hit@5 \| Hit@10 \| MRR \| NDCG \| Note | Results | `data/results/thesis_tables/best_per_entity.csv` | **Data exists** | `tab:best_representation` |
| Tab 3.8 | Extracted RDF Predicates Per Entity Type | Detailed predicate inventory (can merge with Tab 3.1 or use as appendix) | RQ1 | §3.4.2 | Columns: Entity type \| Predicate category \| Predicate URIs \| Resolved field | Methodology | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` | No — compile manually | `tab:predicate_inventory` |
| Tab 3.9 | Retrieval Method Design Matrix | Catalogues all six retrieval methods with their signal types | RQ2 | §3.5.1 | Columns: Method name \| Family \| Signal type \| Input \| Symbolic overlay \| Design rationale | Methodology | `src/retrieval/` | No — compile manually | `tab:retrieval_method_matrix` |
| Tab 3.10 | Retrieval Method Comparison (Main) | Reports retrieval quality for all 6 methods | RQ2 | §3.5.7 | Columns: Method \| Hit@1 \| Hit@5 \| Hit@10 \| MRR \| NDCG \| Δ NDCG vs. baseline; best per metric bolded | Results | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` | **Data exists** | `tab:retrieval_main` |
| Tab 3.11 | Retrieval NDCG by Difficulty | Shows which methods help for hard questions | RQ2 | §3.5.7 | Columns: Method \| Easy NDCG \| Medium NDCG \| Hard NDCG \| Unknown NDCG | Results | `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` | **Data exists** | `tab:retrieval_difficulty` |
| Tab 3.12 | Retrieval NDCG by Entity Type | Shows method performance across paper/dataset/model | RQ2 | §3.5.7 | Columns: Method \| Paper NDCG \| Dataset NDCG \| Model NDCG | Results | `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv` | **Data exists** | `tab:retrieval_entity_type` |
| Tab 3.13 | Precision-Recall Trade-off | Illustrates Hit@1 vs. Hit@10 gap per method | RQ2 | §3.5.7 | Columns: Method \| Hit@1 \| Hit@10 \| Gap (Hit@10 − Hit@1) \| Interpretation | Results | `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` | **Data exists** | `tab:precision_recall_tradeoff` |
| Tab 3.14 | Canonical Entity Record Fields | Documents the canonical record schema for all three entity types | RQ1 | §3.4.2 | Columns: Entity type \| Field name \| Source predicate(s) \| Type \| Example | Methodology | `build_*_records.py` scripts | No — compile manually | `tab:canonical_record_schema` |

---

## 11. Source-to-Section Traceability

| Thesis Section | Main Claim | Source Code Files | Config Files | Result Files | Docs Files | Citation Needed? | TODO Before Final Writing |
|---|---|---|---|---|---|---|---|
| §3.1 | Three-stage pipeline architecture | `src/retrieval/pipeline.py`, `src/post_retrieval/pipeline/context_builder.py` | — | — | `docs/post_retrieval/thesis_overview.md`, `CLAUDE.md` | TODO cite: RAG | None |
| §3.2 | MLSea = 26,606,202 triples, 6.4 GB | `data/raw/pwc_1.nt` (line count) | — | — | `README.md` | TODO cite: RDF/KG | Verify exact line count with `wc -l pwc_1.nt` |
| §3.2 | Three entity types, predicate vocabularies | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` | — | — | `docs/post_retrieval/pre_retrieval_methodology.md` | TODO cite: RDF vocabularies | Compile Table 3.1 from source code |
| §3.3 | 280 questions, 265 answerable, 15 unanswerable | `data/questions/ml_questions_dataset.json` | — | — | — | — | Extract full question type taxonomy from JSON |
| §3.3 | Closed-world retrieval evaluation design | `src/pre_retrieval/shared/evaluate_retrieval.py` | — | — | `docs/post_retrieval/Evaluation_Strategy.md` | TODO cite: information retrieval evaluation | Document difficulty assignment criteria |
| §3.4.1 | Raw RDF triples cannot be embedded as coherent units | `src/pre_retrieval/papers/raw/build_paper_records.py` | — | — | `docs/post_retrieval/pre_retrieval_methodology.md` | TODO cite: RDF/KG, dense retrieval | None |
| §3.4.2 | Two-pass streaming extraction design | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`, `shared/utils.py` | — | — | — | — | Compile Table 3.1 and Table 3.14 |
| §3.4.3 | 200k paper subset with gold-first inclusion | `src/pre_retrieval/papers/raw/build_curated_subset.py` | `config/pre_retrieval_config.json` (max_papers: 200000) | `data/intermediate/raw_papers/papers_subset_200k.jsonl` | — | — | Document total size of `papers_master.jsonl` |
| §3.4.4 | 6 paper representation strategies with field contents and char limits | All `src/pre_retrieval/papers/chunking/build_*.py` | `config/pre_retrieval_config.json` | `data/intermediate/representations/papers/` | — | — | Compile Table 3.3 |
| §3.4.4 | 4 dataset representation strategies | `src/pre_retrieval/datasets/chunking/` | `config/pre_retrieval_config.json` | `data/intermediate/representations/datasets/` | — | — | Verify exact char limits from scripts |
| §3.4.4 | 4 model representation strategies | `src/pre_retrieval/models/chunking/` | `config/pre_retrieval_config.json` | `data/intermediate/representations/models/` | — | — | Verify exact char limits from scripts |
| §3.4.5 | `all-MiniLM-L6-v2`, 384-dim, L2-norm, cosine | `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py` | `config/pre_retrieval_config.json` | — | — | TODO cite: SBERT/SentenceTransformers, MTEB | Add MTEB score or cite benchmark |
| §3.4.5 | 18 Chroma collections, 8.2 GB, HNSW cosine | `src/pre_retrieval/shared/vector_store.py` | `config/pre_retrieval_config.json` | `data/intermediate/chroma/chroma.sqlite3` | — | TODO cite: ChromaDB, HNSW | Verify 18 collection count from ChromaDB |
| §3.4.6 | Hit@k, MRR, NDCG formulas | `src/pre_retrieval/shared/evaluate_retrieval.py`, `src/retrieval/result_schema.py` | — | — | — | TODO cite: NDCG/MRR/Hit@k | None — formulas verified in code |
| §3.4.7 | Paper best = `enriched_metadata` NDCG 0.8225 | `data/results/thesis_tables/full_comparison.csv` | — | `data/results/pre_retrieval_results/paper_results/enriched_metadata/` | — | — | Fill Hit@5 for all paper representations from results JSON |
| §3.4.7 | Dataset best = `dataset_title_only` NDCG 0.3822 | `data/results/thesis_tables/best_per_entity.csv` | — | `data/results/pre_retrieval_results/dataset_results/dataset_title_only/` | — | — | None |
| §3.4.7 | Model best = `model_predicate_filtered` NDCG 0.8750 | `data/results/thesis_tables/best_per_entity.csv` | — | `data/results/pre_retrieval_results/model_results/model_predicate_filtered/` | — | — | None |
| §3.5.1 | Pipeline fully offline, no SPARQL at inference | `src/retrieval/pipeline.py`, `src/retrieval/dense_baseline.py` | `src/retrieval/config.py` | — | `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/README.md` | — | None |
| §3.5.2 | `pure_semantic_dense` NDCG 0.7337 | `src/retrieval/dense_baseline.py` | `src/retrieval/config.py` | `data/results/retrieval/pure_semantic_dense/metrics.json` | — | — | None — verified |
| §3.5.3 | `hybrid_type_filtering` = baseline (no-op) | `src/retrieval/filtering.py` | — | `data/results/retrieval/hybrid_type_filtering/metrics.json` | — | — | None |
| §3.5.3 | `hybrid_type_onehop_filtering` NDCG 0.7351 | `src/retrieval/filtering.py` (`_onehop_richness()`) | — | `data/results/retrieval/hybrid_type_onehop_filtering/metrics.json` | — | — | None |
| §3.5.3 | `hybrid_predicate_aware_filtering` NDCG 0.7375, Hit@1 0.6868 | `src/retrieval/filtering.py` (`_boost_by_predicate()`) | — | `data/results/retrieval/hybrid_predicate_aware_filtering/metrics.json` | — | — | Document the full question_type → predicate mapping |
| §3.5.4 | RRF k=60 fusion groups | `src/retrieval/rrf.py` | `src/retrieval/config.py` (BEST_REPRESENTATIONS, RRF groups) | `data/results/retrieval/optional_rrf_fusion/metrics.json` | — | TODO cite: RRF (Cormack et al. 2009) | None |
| §3.5.4 | `optional_rrf_symbolic` NDCG 0.7434, dataset gain +0.0823 | `src/retrieval/rrf.py` | — | `data/results/retrieval/optional_rrf_symbolic/metrics.json`; `retrieval_by_entity_type_ndcg.csv` | — | — | None |
| §3.5.5 | Top-k = 10 | `src/retrieval/config.py` | — | — | — | — | TODO: justify k=10 (standard default or ablation) |
| §3.5.7 | NDCG range 0.7337–0.7434 across all methods | — | — | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` | — | — | Format Table 3.10 with delta column |
| §3.6 | Hit@10 ceiling = 0.78–0.82 | — | — | `retrieval_main_comparison.csv` | — | — | None |

---

## 12. Citation Needs for Methodology

> Do not search the web. Do not invent bibliography entries. These are placeholder labels to be resolved before final chapter submission.

| Topic | Placeholder | Where Needed in Chapter | Note |
|---|---|---|---|
| Retrieval-Augmented Generation | `TODO cite: RAG` | §3.1 (overview), §3.4.1 | Cite original RAG paper (Lewis et al., 2020 or similar) |
| RDF / Knowledge Graphs | `TODO cite: RDF/KG` | §3.2, §3.4.1 | Cite W3C RDF standard and/or KG survey |
| Knowledge graph verbalization / graph-to-text | `TODO cite: KG verbalization` | §3.4.1 (motivation for entity-centric chunk) | May cite graph-to-text or KG linearisation literature |
| Dense retrieval | `TODO cite: dense retrieval` | §3.5.1, §3.5.2 | Cite DPR (Karpukhin et al., 2020) or dense retrieval survey |
| SBERT / SentenceTransformers | `TODO cite: SBERT/SentenceTransformers` | §3.4.5 (embedding generation), §4.10 | Cite Reimers & Gurevych, 2019 |
| `all-MiniLM-L6-v2` model | `TODO cite: all-MiniLM-L6-v2` | §3.4.5 | Cite HuggingFace model card or training paper |
| MTEB benchmark | `TODO cite: MTEB` | §3.4.5 (model choice justification) | Cite Muennighoff et al., 2022 |
| Vector stores | `TODO cite: vector stores` | §3.4.5 (ChromaDB discussion) | Cite ChromaDB documentation or a vector store survey |
| Approximate nearest neighbour search | `TODO cite: ANN search` | §3.4.5 (HNSW context) | Cite ANN survey or seminal paper |
| HNSW | `TODO cite: HNSW` | §3.4.5 (ChromaDB indexing) | Cite Malkov & Yashunin, 2018 |
| Hybrid retrieval | `TODO cite: hybrid retrieval` | §3.5.1 (retrieval stage intro) | Cite hybrid sparse+dense retrieval papers |
| Symbolic retrieval / knowledge-aware retrieval | `TODO cite: symbolic retrieval` | §3.5.3 (hybrid methods intro) | Cite KG-augmented retrieval literature |
| Reciprocal Rank Fusion | `TODO cite: RRF` | §3.5.4 (RRF formula) | Cite Cormack, Clarke & Buettcher, 2009 |
| Hit@k / MRR / NDCG | `TODO cite: NDCG/MRR/Hit@k` | §3.4.6, §3.5.6 (metric definitions) | Cite IR evaluation textbook (Manning et al., 2008) or TREC evaluation papers |

---

## 13. Methodological Limitations and Validity Considerations

This section discusses limitations specific to RQ1 and RQ2. Post-retrieval answer evaluation limitations are not included.

---

**1. Closed-world retrieval assumption**
The evaluation assumes that the gold target entity is always present in the retrieval corpus (enforced by the gold-first curation rule). This is a closed-world assumption: the system is never tested on questions where the correct answer is absent from the corpus. Closed-world evaluation overestimates performance compared to an open-world deployment where the gold entity may not be indexed. The 200k paper subset amplifies this: any question whose gold paper falls outside the subset would be trivially unanswerable, so the curation rule eliminates this case at the cost of evaluation realism.

**2. Single gold target per question**
Each evaluation question has exactly one `target_entity_iri`. In reality, a question like "What datasets are used for sentiment analysis?" may have multiple valid answers. The single-gold-target design means that a retrieval method that returns multiple highly relevant entities beyond the designated gold is not rewarded. MRR and NDCG are designed for single-relevance evaluation; they do not capture multi-relevant retrieval quality. This is a known limitation of the evaluation design, acceptable at the thesis scale but not representative of production QA settings.

**3. 200k paper subset**
The paper corpus is limited to 200,000 papers, a fraction of the full MLSea paper set. **TODO:** Document the exact size of `papers_master.jsonl`. If the full corpus contains substantially more papers, the 200k subset may not be representative, and retrieval quality metrics may not generalise to the full-corpus setting. The subset may be disproportionately skewed toward high-profile papers with many linked nodes (since those are more likely to appear as gold targets for evaluation questions).

**4. Gold-first corpus curation**
The gold-first curation rule guarantees gold target inclusion but also introduces a subtle bias: the evaluation corpus contains all gold entities unconditionally, which means that the corpus is not a random sample of the MLSea paper population. If gold papers are systematically more richly annotated than average papers, the retrieval environment is easier than it would be in a random-sample corpus.

**5. Sparse dataset metadata**
The NDCG ceiling for datasets (0.3822 for the best representation) reflects a fundamental data quality problem in the source KG, not a failure of the retrieval method. Many MLSea dataset entities have only a title and no further metadata. This sparsity limits the discriminative power of any representation strategy and means that approximately 71.8% of dataset questions fail to rank the gold entity at position 1. This is an inherent property of the knowledge graph, not an addressable limitation within the pre-retrieval stage.

**6. Dependency on `question_type` labels for predicate-aware filtering**
The `hybrid_predicate_aware_filtering` method requires a `question_type` label for each question to determine which metadata field to boost. The quality of this method therefore depends on the completeness and consistency of the question type taxonomy. Questions with unmapped or unknown types fall back to the semantic ranking, receiving no predicate boost. **TODO:** Verify the full question_type taxonomy from `data/questions/ml_questions_dataset.json` and document which types have defined predicate mappings.

**7. Limitations of title-only representations**
For datasets, `dataset_title_only` is the best representation, but it is inherently limited: if two datasets share similar titles or if a question uses paraphrased phrasing that does not closely match the title, retrieval will fail. Title-only representations provide no fallback vocabulary.

**8. Limitations of dense embeddings for short or ambiguous titles**
Short titles (e.g., "CNN", "Transformer") produce embedding vectors that are close to many other entities with similar names. The 384-dimensional embedding space may not provide sufficient discriminative capacity for entities whose titles are highly polysemous or common terms in the ML domain. Model entities are particularly affected: model names are often generic.

**9. Limitations of symbolic filtering when metadata is missing**
The one-hop richness filter and predicate-aware filter rely on the presence of structured metadata fields (`tasks`, `datasets`, `implementations`, etc.) in the candidate record. For entities with sparse metadata (primarily datasets and some models), these fields are empty, and the symbolic signals provide no boost. The result is that symbolic filtering benefits well-annotated entities more than sparse ones — the entities that most need additional retrieval signal receive the least.

**10. Limitations of one-hop richness as a relevance proxy**
Graph connectivity density (number of populated metadata fields) is a proxy for relevance, not a direct measure. A paper with many linked datasets, tasks, and methods may be highly connected but irrelevant to a specific question. The richness signal does not incorporate question intent. Its marginal improvement (+0.0013 NDCG) confirms that it is a weak proxy.

**11. Limitations of RRF when representations disagree**
RRF assumes that an entity consistently ranked highly across multiple representations is more likely to be relevant. When representations capture different semantic facets, they may produce systematically different rankings for the same entity — one representation may rank the correct entity highly while another ranks it poorly. RRF averages these disagreements by rank, which can depress the rank of the correct entity if multiple representations disagree. This is reflected in the reduced Hit@1 for RRF methods compared to the single-best-representation baseline.

**12. Top-10 cutoff and downstream impact**
The evaluation computes metrics only for gold entities found within the top-10 candidates. Questions where the gold entity ranks below 10 contribute a score of 0 to all metrics. The top-10 cutoff also defines the hard recall ceiling for post-retrieval: approximately 19–22% of questions fail to include the gold entity in the top-10, and no downstream re-ranking method can recover these. **TODO:** Calculate the exact percentage of questions with gold outside top-10 for each method from `data/results/retrieval/*/metrics.json`.

**13. Why post-retrieval cannot recover a gold entity absent from the top-10**
The post-retrieval stage receives only the top-10 candidates as its input. If the gold entity is not in this set, it is not available for re-ranking, context assembly, or answer generation. The Hit@10 value therefore represents an absolute performance ceiling that no post-retrieval method can exceed. This is a structural property of the pipeline architecture, not a limitation of any specific method.

**14. Embedding model not fine-tuned on MLSea domain**
The `all-MiniLM-L6-v2` model is used without domain adaptation or fine-tuning on MLSea question-entity pairs. A model fine-tuned on contrastive pairs from the MLSea domain (e.g., using a question and its gold entity chunk as a positive pair) would likely achieve higher retrieval quality. The current evaluation represents an out-of-the-box zero-shot embedding quality.

**15. TODO — Unanswerable question criterion**
15 questions are marked `is_answerable = false`. The criterion for this label is not documented in the repository. **TODO:** Inspect these 15 questions in `data/questions/ml_questions_dataset.json` and document what makes them unanswerable — no entity in the graph, ambiguous gold target, or genuinely no correct answer.

---

## 14. What Not to Write

> For each domain, a weak example and a polished academic alternative are shown.

---

**RDF Extraction**

Weak:
> "I extracted papers from the RDF file."

Better:
> "Paper entity records are extracted from the MLSea N-Triples file through a two-pass streaming parse. The first pass collects all predicate-object pairs for subjects matching the paper entity prefix; the second pass resolves linked-node IRIs to human-readable labels using a pre-built label cache, enabling entity records to contain descriptive vocabulary rather than opaque graph identifiers."

---

**Chunking**

Weak:
> "I created chunks for each paper using different strategies."

Better:
> "For each paper entity, six representation strategies are constructed by selecting different subsets of the canonical entity record's fields, applying character-length truncation, and concatenating the selected values into a flat UTF-8 string. The strategies are designed to explore the trade-off between representation specificity — the degree to which the chunk captures ML-domain semantics — and representation noise — the degree to which irrelevant fields dilute the embedding signal."

---

**Embedding**

Weak:
> "I used MiniLM to embed chunks."

Better:
> "The textual entity representations are encoded into a shared semantic vector space using `sentence-transformers/all-MiniLM-L6-v2`, enabling natural-language questions and KG-derived entity chunks to be compared through cosine similarity. Embeddings are L2-normalised prior to storage, ensuring that cosine similarity is equivalent to the dot product of the vector representations."

---

**Pre-Retrieval Evaluation**

Weak:
> "I evaluated the representations using NDCG."

Better:
> "The retrieval quality of each representation strategy is quantified using Normalised Discounted Cumulative Gain (NDCG), which discounts the contribution of the correct entity by its rank position and serves as the primary metric throughout this evaluation. NDCG is chosen over Hit@1 because it provides a graded assessment of ranking quality: placing the gold entity at rank 1 is rewarded more than placing it at rank 5, reflecting the downstream importance of top-ranked candidates for context construction."

---

**Dense Retrieval**

Weak:
> "The dense retrieval baseline just uses cosine similarity."

Better:
> "The `pure_semantic_dense` baseline applies the cosine-similarity ranking from the pre-retrieval stage without modification. It establishes the performance achievable by semantic embedding alone, given the best-per-entity-type representation identified in §3.4.7. All subsequent retrieval methods are evaluated relative to this baseline; a method that does not improve upon `pure_semantic_dense` fails to justify its added complexity."

---

**Hybrid Retrieval**

Weak:
> "I added some extra signals to improve retrieval."

Better:
> "The hybrid retrieval methods augment the semantic ranking with symbolic signals derived from the KG's structural metadata. These signals — entity-type consistency, one-hop graph connectivity density, and question-type-specific predicate presence — are pre-computed from the candidate records and applied as re-ranking criteria without any live graph query. The hybrid design tests whether the structured, relational information encoded in the MLSea knowledge graph provides retrieval value beyond what the dense embedding captures."

---

**RRF**

Weak:
> "I combined multiple rankings using RRF."

Better:
> "Reciprocal Rank Fusion (RRF) is applied to aggregate candidate rankings from multiple pre-retrieval representation strategies for the same entity type. The RRF score for each candidate is computed as the sum of 1/(60 + rank_i) across all representations in which it appears, where k=60 is the standard smoothing constant that prevents the top ranks from dominating the aggregate score (TODO cite: Cormack et al., 2009). This aggregation rewards candidates that are consistently ranked highly across multiple independent representations."

---

**Retrieval Interpretation**

Weak:
> "The results show that all methods perform similarly."

Better:
> "The six retrieval methods produce NDCG values in the narrow range 0.7337–0.7434, a spread of 0.0097 NDCG points. This tight clustering indicates that, given the semantic representations selected in §3.4.7, the symbolic and structural signals available in the pre-computed candidate metadata provide only marginal retrieval value. The binding constraint is not the candidate generation strategy but the quality of the entity-centric semantic embedding: once the representation captures the relevant semantic content, the cosine-similarity ranking is largely determined by the embedding geometry, and symbolic post-processing yields diminishing returns."

---

## 15. Final Checklist Before Writing Chapter 3

- [ ] **RQ1 is fully addressed.** Pre-retrieval stage motivation, methods, evaluation, and findings are present in Chapter 3 (§3.4).
- [ ] **RQ2 is fully addressed.** Retrieval stage objective, all six methods, evaluation, results, and interpretation are present in Chapter 3 (§3.5).
- [ ] **RQ3 is not expanded.** Post-retrieval evaluation appears only as a boundary note in §3.6. No RQ3 experiments, metrics, or evaluation design appear in Chapter 3.
- [ ] **RQ2 wording is aligned with implementation.** Chapter 3 describes the retrieval methods in terms of offline metadata signals, not live SPARQL queries. The phrase "SPARQL-based filtering" does not appear in the retrieval stage description.
- [ ] **All metrics are verified.** Every numerical value in Chapter 3 is traceable to a file in `data/results/` or verified from source code. No value is stated from memory alone.
- [ ] **Question type taxonomy is documented.** The full list of unique `question_type` values and their counts has been extracted from `data/questions/ml_questions_dataset.json` and appears in Table 3.2.
- [ ] **All TODO values are resolved.** Every `TODO:` marker in this plan has been addressed before submission. In particular: difficulty assignment criteria, full pre-retrieval metric table (Hit@5/10 for all representations), full question type taxonomy, size of `papers_master.jsonl`, and k=10 justification.
- [ ] **All 8 figures are created.** Figures 3.1–3.8 are generated or drawn and saved to `data/results/thesis_figures/`. LaTeX labels are assigned per Table 10.
- [ ] **All 14 tables are formatted.** Tables 3.1–3.14 are formatted for LaTeX or Word. Existing CSV files are converted; manual compilation tables (3.1, 3.2, 3.3, 3.4, 3.8, 3.9, 3.14) are written.
- [ ] **All citation placeholders are resolved.** Every `TODO cite:` marker in this plan has been replaced with an actual bibliography entry in the thesis reference list.
- [ ] **Methodology/results separation is maintained.** No metric values appear in the method description subsections (§3.4.2–3.4.5, §3.5.2–3.5.5). Metric values appear only in §3.4.7 (pre-retrieval results) and §3.5.7 (retrieval results).
- [ ] **Every design choice is justified.** Character limits, k=10, k=60 (RRF), batch size 64, `all-MiniLM-L6-v2`, 200k subset, NDCG as primary metric — each has a rationale in the text.
- [ ] **Every claim is linked to a source file or result file.** No claim about system behaviour or metric values is made without a source file citation or result file reference.
- [ ] **Terminology is consistent throughout.** The terms "chunk", "entity-centric representation", "canonical entity record", "candidate", "semantic ranking", "symbolic signal", "hybrid retrieval", and "top-10" are used consistently with the definitions in §3.1 and §6.
- [ ] **Post-retrieval is mentioned only as a boundary/future stage.** §3.6 describes what retrieval delivers and what post-retrieval needs, but does not develop RQ3 experiments, evaluation metrics, or answer generation design.

---

*End of Methodology Chapter — Writing Execution Plan v2 (Pre-Retrieval and Retrieval Only)*
