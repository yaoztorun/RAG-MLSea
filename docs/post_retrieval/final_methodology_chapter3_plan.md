# Chapter 3 — Final Methodology Plan
## Scope: Pre-Retrieval and Retrieval Stages (RQ1 and RQ2)

**Thesis:** Retrieval-Augmented Generation over Machine Learning Knowledge Graphs  
**Institution:** KU Leuven  
**Target Chapter:** Chapter 3 — Methodology  
**Scope:** Pre-retrieval entity representation construction and retrieval candidate ranking  
**Version:** final — supersedes `methodology_writing_execution_plan.md` and `methodology_writing_execution_plan_v2_pre_retrieval_retrieval_only.md`  
**All metrics and implementation details verified against repository files unless marked TODO.**

---

## Table of Contents

1. [Final Chapter 3 Purpose and Narrative](#1-final-chapter-3-purpose-and-narrative)
2. [Exact Chapter 3 Outline with Writing Purpose](#2-exact-chapter-3-outline-with-writing-purpose)
3. [Detailed Section-by-Section Writing Plan](#3-detailed-section-by-section-writing-plan)
4. [Chapter 3 Figures and Tables Plan](#4-chapter-3-figures-and-tables-plan)
5. [Algorithm Boxes for Chapter 3](#5-algorithm-boxes-for-chapter-3)
6. [Chapter 3 Source-to-Section Traceability](#6-chapter-3-source-to-section-traceability)
7. [Chapter 3 Citation Needs](#7-chapter-3-citation-needs)
8. [Chapter 3 Writing Checklist](#8-chapter-3-writing-checklist)
9. [Remaining TODOs](#9-remaining-todos)

---

## 1. Final Chapter 3 Purpose and Narrative

### 1.1 The Purpose of Chapter 3

Chapter 3 presents the methodology of the KG-RAG system developed in this thesis. The system is designed to answer natural-language questions about machine-learning entities by retrieving relevant entity records from the MLSea knowledge graph. The complete pipeline comprises three sequential stages: pre-retrieval, retrieval, and post-retrieval. Chapter 3 develops the design and implementation of the first two stages in full, and introduces the post-retrieval stage as the natural continuation of the pipeline that consumes the retrieval output.

The chapter does not read as a code walkthrough or a sequence of implementation steps. It reads as a methodological argument: every design decision is motivated, every choice between alternatives is justified, and every component connects logically to the next. The empirical validation of these choices — the metric values, the comparison tables, the answers to RQ1 and RQ2 — belongs in Chapter 4. Chapter 3 establishes what was built and why; Chapter 4 establishes how well it worked.

### 1.2 The Story Chapter 3 Must Tell

**Why MLSea cannot be used as raw RDF triples for dense retrieval.**
The MLSea knowledge graph encodes the machine-learning literature as a collection of discrete RDF triples, each stating a single predicate-object relationship for a subject entity. While this representation is machine-interpretable and supports structured querying, it is fundamentally unsuited to dense vector retrieval. A dense retrieval system expects corpus units to be coherent, semantically self-contained passages. A raw RDF triple — such as `<pwc/scientificWork/bert> <mlso:hasTaskType> <pwc/task/question-answering>` — is neither coherent nor self-contained: the subject is an opaque IRI, the predicate is a namespace token, and the object is a linked node that must itself be resolved to a human-readable label. The information needed to answer a question like "What tasks does the BERT paper address?" is distributed across dozens of triples, none of which individually constitutes a retrievable unit.

**Why entity-centric chunk construction is necessary.**
The pre-retrieval stage exists to bridge this structural gap. By aggregating all relevant triples for a single entity into a single textual string — an *entity-centric chunk* — the system creates corpus units that are semantically coherent, contextually complete, and directly comparable to natural-language questions via cosine similarity. This aggregation also resolves the two-level indirection problem inherent in RDF: linked-node IRIs are resolved to their human-readable labels through a second extraction pass, so the final chunk contains terms such as "Question Answering" and "SQuAD" rather than anonymous graph identifiers.

**Why multiple chunking strategies are compared.**
Entity-centric construction raises a non-trivial design question: which fields should be included in the chunk, and to what depth should linked entities be traversed? Including too few fields produces sparse representations that may lack the vocabulary needed to match question phrasing. Including too many fields risks diluting the most distinctive signal with noisy or irrelevant predicates. Different entity types compound this challenge: a paper has a rich abstract that anchors its semantic identity, whereas a dataset may have only a title and sparse metadata. The thesis therefore evaluates 14 representation strategies — six for papers, four for datasets, four for models — spanning a spectrum from minimal to maximal field coverage. This comparison is the empirical backbone of RQ1.

**Why the best pre-retrieval representation per entity type becomes the retrieval input.**
The outcome of RQ1 is not a single universal best representation, but an entity-type-specific optimum: `enriched_metadata` for papers, `dataset_title_only` for datasets, and `model_predicate_filtered` for models. These selections are determined by evaluation against the question set (reported in Chapter 4) and then used as fixed inputs for the retrieval stage. Fixing the representation ensures that the retrieval comparison begins from the strongest available semantic foundation.

**Why retrieval methods are compared after representation selection.**
The retrieval stage asks a distinct question: given the best possible semantic representation, can additional signals drawn from the knowledge graph's structural metadata improve the ranked candidate list? By fixing the representation and varying the retrieval method, the thesis isolates the contribution of symbolic and structural signals over and above pure semantic similarity. This design separates two sources of retrieval quality that are often conflated in the RAG literature.

**Why hybrid symbolic-semantic retrieval is meaningful for KG-based RAG.**
A knowledge graph encodes explicit symbolic structure: each entity has typed relationships, typed predicates, and metadata fields that carry factual semantics. A hybrid retrieval method that exploits this structure — through entity-type consistency filtering, metadata field richness boosting, question-type-to-predicate alignment, or multi-representation fusion — can in principle achieve precision improvements inaccessible to pure semantic similarity. The thesis evaluates six such methods and their empirical relationship to the dense baseline.

**How Chapter 3 ends.**
Chapter 3 concludes by specifying the output of the retrieval stage: a ranked list of at most ten candidate entities per question. The retrieved candidate set constitutes the evidence base for the downstream post-retrieval stage, which is introduced as the third component of the complete KG-RAG pipeline. The post-retrieval stage receives this candidate set and proceeds with context assembly, potential re-ranking, and answer generation. The detailed evaluation of retrieval method performance is presented in Chapter 4.

### 1.3 Separation of Methodology and Results

Separating methodology (Chapter 3) from results (Chapter 4) improves the thesis in three ways. First, it allows a reader to evaluate the soundness of the design independently of its empirical performance — a methodology can be well-motivated even if results are modest, or empirically strong even if its motivation is underspecified. Second, it prevents premature anchoring of design decisions to observed metrics: the methodology chapter describes what was built before revealing how it performed. Third, it mirrors standard practice in the information retrieval and NLP literature, where system description and experimental results are treated as distinct contributions.

---

## 2. Exact Chapter 3 Outline with Writing Purpose

### 2.1 Mandatory Layout

```
3.   Methodology

3.1  Methodological Overview

3.2  Dataset and Knowledge Graph Source
     3.2.1  The MLSea Knowledge Graph
     3.2.2  Entity Types and RDF Structure
     3.2.3  Machine-Learning Domain Metadata in MLSea

3.3  Question Set and Retrieval Task Definition
     3.3.1  Question Dataset
     3.3.2  Gold Target Entity Design
     3.3.3  Retrieval Task Formulation

3.4  Pre-Retrieval Phase: Entity Representation Construction
     3.4.1  Motivation for Pre-Retrieval Representations
     3.4.2  RDF Extraction and Linked-Entity Label Resolution
     3.4.3  Canonical Entity Record Construction
     3.4.4  Corpus Curation and Subset Selection
     3.4.5  Entity-Centric Chunk Construction
            3.4.5.1  Paper Representations
            3.4.5.2  Dataset Representations
            3.4.5.3  Model Representations
     3.4.6  Embedding Generation and Vector Indexing
     3.4.7  Pre-Retrieval Output

3.5  Retrieval Phase: Candidate Entity Ranking
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
     3.5.6  Retrieval Output
```

Post-retrieval is introduced as the third pipeline stage in §3.1 and referenced as the downstream consumer of the retrieval output in §3.5.6. No separate numbered post-retrieval section is added.

### 2.2 Section-by-Section Purpose Table

| Section | Title | Purpose | What to Write | What NOT to Include | Supporting Files | Suggested Figure/Table | Content Type |
|---|---|---|---|---|---|---|---|
| 3.1 | Methodological Overview | Orient the reader; introduce the complete three-stage KG-RAG pipeline | One-page overview: pre-retrieval → retrieval → post-retrieval; terminology commitment; entity-centric design principle; RQ alignment | Detailed metric values; result tables; post-retrieval implementation details | `docs/post_retrieval/thesis_overview.md`, `CLAUDE.md` | Figure 3.1 (pipeline overview) | Methodological overview |
| 3.2.1 | The MLSea Knowledge Graph | Describe the raw data source — the N-Triples RDF file, its scale, and its format | 6.4 GB file, 26,606,202 triples, N-Triples format, entity IRI prefixes, Papers with Code provenance | Chunk construction; embedding; retrieval results | `data/raw/pwc_1.nt`, `src/pre_retrieval/papers/raw/build_paper_records.py` | Figure 3.2 (RDF entity structure example) | Dataset description |
| 3.2.2 | Entity Types and RDF Structure | Describe the three entity types and their RDF structure | Papers, datasets, models; IRI namespace per type; triple structure; subject/predicate/object semantics | Representation strategies; metric values | `CLAUDE.md` entity types section, `pre_retrieval_methodology.md` | Table 3.1 (entity types and roles) | Dataset description |
| 3.2.3 | Machine-Learning Domain Metadata in MLSea | Describe the ML-domain predicates that carry task, dataset, method, and metric information | `mlso:hasTaskType`, `dcat:keyword`, linked-entity structure; annotation density variation; sparse vs. rich entities | Chunk construction detail | `build_paper_records.py`; predicate lists from v2 plan §4.1 | Table 3.2 (predicate-to-field mapping) | Dataset description |
| 3.3.1 | Question Dataset | Describe the evaluation question set | 280 questions, 265 answerable, 15 unanswerable; field schema; `question_type`, `target_entity_iri`, `difficulty`, `is_answerable` | Gold target selection rationale (§3.3.2); retrieval task definition (§3.3.3) | `data/questions/ml_questions_dataset.json` | Table 3.3 (question dataset schema) | Dataset description |
| 3.3.2 | Gold Target Entity Design | Explain how the ground-truth answer entity is defined | `target_entity_iri` as single gold IRI per question; closed-world assumption; IRI normalisation before matching | How matching is computed (metric formulas belong in Chapter 4) | `data/questions/ml_questions_dataset.json`, `CLAUDE.md` | Table 3.4 (gold target entity design) | Retrieval task definition |
| 3.3.3 | Retrieval Task Formulation | Formally define the retrieval task | Input: natural-language question; output: ranked list of ≤10 entity IRIs; success: gold IRI in top-k; k=10 as default; segmentation plan (by entity type, difficulty, question type) | Metric formulas; numeric results | `src/retrieval/config.py` (DEFAULT_TOP_K=10) | Table 3.5 (retrieval task formulation) | Retrieval task definition |
| 3.4.1 | Motivation for Pre-Retrieval Representations | Justify why raw RDF cannot serve as corpus units | Two problems: lexical incoherence (IRI strings); two-level indirection (linked nodes); entity-centric construction as the solution | Any metric values; result interpretation | `docs/post_retrieval/pre_retrieval_methodology.md`; v2 plan §2 narrative | Figure 3.3 (RDF triples → entity chunk) | Technical design |
| 3.4.2 | RDF Extraction and Linked-Entity Label Resolution | Explain the two-pass streaming extraction | Pass 1: entity subjects + predicate-object pairs; Pass 2: linked-node label resolution via `rdfs:label`, `foaf:name`, `dcterms:title`; node label cache; IRI fallback | Chunk construction; character limits | `src/pre_retrieval/papers/raw/build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`, `shared/utils.py` | Figure 3.4 (two-pass extraction); Table 3.2 | Technical design |
| 3.4.3 | Canonical Entity Record Construction | Describe the unified structured record for each entity | Predicate-to-field mapping; canonical field schema per entity type; deduplication; priority order for multi-predicate fields | Chunk construction; evaluation | All three `build_*_records.py` files | Table 3.6 (canonical entity record fields) | Technical design |
| 3.4.4 | Corpus Curation and Subset Selection | Explain the gold-first 200k paper subset | Gold-target inclusion guarantee; 200k cap from `config/pre_retrieval_config.json`; remaining capacity filled in file order; output path | Why 200k was chosen over alternatives (keep brief); full corpus size (TODO) | `src/pre_retrieval/papers/raw/build_curated_subset.py`, `config/pre_retrieval_config.json` | Table 3.7 (corpus construction summary) | Technical design |
| 3.4.5.1 | Paper Representations | Describe all six paper representation strategies | Field content and character limits per strategy; design rationale per strategy; contrast between minimal (title_only) and maximal (enriched_metadata) | NDCG values; which performs best (Chapter 4) | `src/pre_retrieval/papers/chunking/build_*.py`, `config/pre_retrieval_config.json` | Table 3.8 (strategy matrix); Figure 3.6 (example chunks) | Technical design |
| 3.4.5.2 | Dataset Representations | Describe all four dataset representation strategies | Field content per strategy; dataset sparsity problem; why annotation density differs from papers | Numeric NDCG results; best-representation conclusion | `src/pre_retrieval/datasets/chunking/` | Table 3.8 (continued) | Technical design |
| 3.4.5.3 | Model Representations | Describe all four model representation strategies | Field content per strategy; model specificity problem; predicate whitelist rationale | Numeric NDCG results; best-representation conclusion | `src/pre_retrieval/models/chunking/` | Table 3.8 (continued) | Technical design |
| 3.4.6 | Embedding Generation and Vector Indexing | Describe embedding model and ChromaDB setup | `all-MiniLM-L6-v2`; 384-dim; L2 normalisation; cosine metric; ChromaDB HNSW; one collection per representation strategy; batch size 64 | Retrieval metrics; comparison of embedders | `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py`, `config/pre_retrieval_config.json` | Figure 3.7 (embedding/indexing workflow); Table 3.9 (indexing config) | Technical design |
| 3.4.7 | Pre-Retrieval Output | Describe what the pre-retrieval stage delivers | Per-question top-10 candidate lists per representation strategy; metric-based selection of best representation per entity type as input to retrieval; output paths | Full metric tables (Chapter 4) | `data/results/thesis_tables/best_per_entity.csv`, `data/results/pre_retrieval_results/` | None required (results in Chapter 4) | Pipeline output |
| 3.5.1 | Retrieval Objective and Design Rationale | Explain the transition from pre-retrieval to retrieval | Fixed best representation as input; retrieval as candidate re-weighting/re-ordering; symbolic signals as complement to semantic similarity; six-method design | Numeric results; method comparison tables | `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/config.py` | Figure 3.8 (retrieval method workflow); Table 3.10 (method design matrix) | Technical design |
| 3.5.2 | Dense Retrieval Baseline | Describe `pure_semantic_dense` | Passthrough of pre-retrieval top-10 for the best representation per entity type; establishes the semantic ranking as the comparison baseline | Numeric NDCG for baseline (Chapter 4) | `src/retrieval/dense_baseline.py` | Part of Table 3.10 | Technical design |
| 3.5.3.1 | Type-Filtered Retrieval | Describe `hybrid_type_filtering` | Entity-type consistency filter using IRI prefix; control method; expected to confirm collection purity | That it is a no-op empirically (Chapter 4) | `src/retrieval/filtering.py:run_type_filtering` | Part of Table 3.10 | Technical design |
| 3.5.3.2 | One-Hop Richness-Boosted Retrieval | Describe `hybrid_type_onehop_filtering` | Type filter + one-hop field richness boost; boost counts non-empty fields from {tasks, datasets, methods, metrics, implementations} plus "Linked Entities" in source text; question-type-agnostic | Numeric results; marginal gain | `src/retrieval/filtering.py:run_hybrid_type_onehop_filtering` | Part of Table 3.10 | Technical design |
| 3.5.3.3 | Predicate-Aware Retrieval | Describe `hybrid_predicate_aware_filtering` | Question-type-to-predicate mapping; boosted candidates sorted before non-boosted; question type taxonomy from `filtering.py` | Hit@1 improvement value (Chapter 4) | `src/retrieval/filtering.py:run_predicate_aware_filtering` | Part of Table 3.10 | Technical design |
| 3.5.4.1 | Reciprocal Rank Fusion | Describe `optional_rrf_fusion` | RRF formula; k=60; fusion groups per entity type (papers: enriched_metadata + predicate_filtered + one_hop; datasets: dataset_title_only + dataset_enriched_metadata; models: model_predicate_filtered + model_enriched_metadata); rank aggregation logic | Hit@10 improvement value (Chapter 4) | `src/retrieval/rrf.py`, `src/retrieval/config.py` | Part of Table 3.10 | Technical design |
| 3.5.4.2 | RRF with Symbolic Filtering | Describe `optional_rrf_symbolic` | RRF first, then type filtering, then predicate-aware boosting; combines fusion breadth with predicate precision | NDCG value (Chapter 4) | `src/retrieval/rrf.py` | Part of Table 3.10 | Technical design |
| 3.5.5 | Top-k Candidate Generation | Explain top-k=10 design and output schema | k=10 default; candidate fields (entity_id, entity_type, rank, score, source_text, metadata); all methods output the same schema; k=10 defines the recall ceiling | Metric formulas; numeric results | `src/retrieval/config.py` (DEFAULT_TOP_K=10), `src/retrieval/pipeline.py`, `src/retrieval/result_schema.py` | Table 3.11 (candidate output schema) | Pipeline output |
| 3.5.6 | Retrieval Output | Describe what the retrieval stage delivers and how it connects to post-retrieval | Per-question ranked candidate list; output paths; the candidate set as input to the subsequent post-retrieval stage; Hit@10 as maximum recoverability ceiling | Post-retrieval implementation; answer generation; any metric values | `src/retrieval/result_schema.py` | Figure 3.9 (top-k output and downstream use) | Pipeline output |

---

## 3. Detailed Section-by-Section Writing Plan

---

### 3.1 Methodological Overview

**Goal:** Introduce the complete three-stage KG-RAG pipeline at a high level and establish the terminology and design principles used throughout the chapter.

**Key ideas:**
- The KG-RAG system developed in this thesis operates in three stages: pre-retrieval, retrieval, and post-retrieval.
- Pre-retrieval converts MLSea RDF entities into dense-retrievable textual representations.
- Retrieval ranks candidate entities by matching embedded question representations against embedded entity representations, using pure dense similarity and several hybrid symbolic-semantic methods.
- Post-retrieval receives the top-k candidate set and proceeds with context assembly, potential re-ranking, and answer generation.
- This chapter develops the pre-retrieval and retrieval stages in full. The post-retrieval stage is introduced as the third pipeline component; its detailed methodology is the subject of subsequent thesis work.
- Chapter 3 is motivated by RQ1 (which representation strategy best supports retrieval?) and RQ2 (does hybrid retrieval outperform pure dense retrieval?).

**Thesis-ready paragraph example:**

> This chapter presents the methodology of a Retrieval-Augmented Generation system designed to answer natural-language questions about machine-learning entities by retrieving relevant records from the MLSea knowledge graph. The pipeline comprises three sequential stages. In the pre-retrieval stage, RDF entities — papers, datasets, and models — are extracted from the raw knowledge graph, converted into canonical entity records, and transformed into entity-centric textual representations that are embedded and indexed in a vector store. In the retrieval stage, each evaluation question is embedded using the same model and matched against the indexed entity representations; candidate entities are ranked by a combination of dense semantic similarity and, in hybrid methods, symbolic signals derived from the knowledge graph's structural metadata. The retrieved candidate set — a ranked list of at most ten entity identifiers per question — constitutes the evidence base for the downstream post-retrieval stage, which assembles this candidate set into a context for answer generation. The detailed methodology of the pre-retrieval and retrieval stages is presented in Sections 3.4 and 3.5 respectively.

**Figure/table:** Figure 3.1 (full pipeline overview) — insert immediately after opening paragraph.

**Source files:** `docs/post_retrieval/thesis_overview.md`, `CLAUDE.md`

**Do NOT include:** metric values; result tables; post-retrieval implementation details; specific NDCG numbers.

---

### 3.2.1 The MLSea Knowledge Graph

**Goal:** Describe the primary data source — the MLSea N-Triples RDF dump — including its scale, format, provenance, and structural characteristics.

**Key ideas:**
- MLSea is a machine-learning knowledge graph built from the Papers with Code dataset.
- Distributed as a single N-Triples file (`data/raw/pwc_1.nt`): 6.4 GB, 26,606,202 triples.
- N-Triples format: one triple per line — `<subject IRI> <predicate IRI> <object IRI or literal> .`
- Entity IRI prefixes:
  - Papers: `http://w3id.org/mlsea/pwc/scientificWork/`
  - Datasets: `http://w3id.org/mlsea/pwc/dataset/`
  - Models: `http://w3id.org/mlsea/pwc/model/`
- The graph is large but flat: no graph nesting, no named graphs.
- Namespace heterogeneity: predicates draw from multiple vocabularies (Dublin Core, FOAF, Schema.org, DCAT, MLSO, FABIO).

**Thesis-ready paragraph example:**

> The primary data source for this thesis is the MLSea knowledge graph, a machine-learning-domain RDF graph derived from the Papers with Code dataset [TODO: cite MLSea]. MLSea encodes research papers, benchmark datasets, and machine-learning models as typed RDF entities linked by structured predicates that express relationships such as task associations, dataset usage, and implementation repositories. The graph is distributed as a single N-Triples file comprising 26,606,202 triples with a total size of 6.4 GB. In the N-Triples serialisation, each triple occupies a single line of the form `<subject> <predicate> <object> .`, where subject and predicate are IRIs and the object is either an IRI or a string literal. Three entity IRI namespaces partition the graph: papers are identified by the prefix `http://w3id.org/mlsea/pwc/scientificWork/`, datasets by `http://w3id.org/mlsea/pwc/dataset/`, and models by `http://w3id.org/mlsea/pwc/model/`.

**Figure/table:** Figure 3.2 (example N-Triples snippet showing subject/predicate/object structure); Table 3.1 (entity types, IRI prefixes, and roles).

**Source files:** `data/raw/pwc_1.nt`; `src/pre_retrieval/papers/raw/build_paper_records.py`

**Move to Chapter 4:** nothing — this is pure dataset description.

---

### 3.2.2 Entity Types and RDF Structure

**Goal:** Describe the three entity types and their structural role in the pipeline.

**Key ideas:**
- Papers: text-rich entities with title, abstract, authors, publication year, tasks, datasets, methods, metrics, and implementations. Identified by `scientificWork/` IRI prefix.
- Datasets: sparse entities; often only a title and brief description, with task and related-paper links. Identified by `dataset/` IRI prefix.
- Models: graph-heavy entities; model names are often generic and their distinguishing characteristics come from task associations and repository links. Identified by `model/` IRI prefix.
- Annotation density varies substantially across entity types and individual entities.
- Linked entities (tasks, methods, metrics) are referenced by IRI; their human-readable labels must be resolved.

**Thesis-ready paragraph example:**

> Within the MLSea graph, three primary entity types are relevant to this thesis. Scientific work entities (referred to throughout as *papers*) correspond to research publications and are identified by IRIs under the `scientificWork/` namespace. They are the most annotation-dense entity type: a well-documented paper entity may carry dozens of predicate-object pairs encoding its title, abstract, authorship, publication year, task associations, benchmark dataset links, and implementation repositories. Dataset entities, identified by the `dataset/` namespace, are considerably sparser: most carry only a label and, in some cases, a brief textual description and a small set of task links. Model entities, identified by the `model/` namespace, are characterised by their association with specific tasks and datasets rather than by natural-language descriptions, making their retrieval informativeness heavily dependent on the quality of linked-entity traversal.

**Figure/table:** Table 3.1 (entity types and roles in the pipeline).

**Source files:** `CLAUDE.md` entity types section; `docs/post_retrieval/pre_retrieval_methodology.md`

---

### 3.2.3 Machine-Learning Domain Metadata in MLSea

**Goal:** Describe the ML-domain predicates that are most relevant to retrieval — the structured links between papers, tasks, datasets, methods, and metrics.

**Key ideas:**
- Key predicates used: `mlso:hasTaskType`, `dcat:keyword`, `dcterms:title`, `fabio:abstract`, `dcterms:creator`, `mlso:hasRelatedImplementation`, `schema:codeRepository`.
- Linked-entity predicates: tasks, datasets, methods, metrics are linked nodes identified by IRI; the link carries the predicate, and the label is stored on the linked node as `rdfs:label` or `foaf:name`.
- Annotation sparsity: not all entities have all predicates; dataset entities in particular may have only `dcterms:title` populated.
- This structural heterogeneity directly motivates entity-type-specific representation strategies.

**Figure/table:** Table 3.2 (RDF predicate-to-field mapping, separate columns for papers, datasets, models).

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py`; `build_dataset_records.py`; `build_model_records.py`

---

### 3.3.1 Question Dataset

**Goal:** Describe the evaluation question set and its field schema.

**Key ideas:**
- 280 total questions; 265 used in metric computation (15 `is_answerable=False` excluded — **TODO: document exclusion criterion**).
- Field schema: `id`, `question` (natural-language string), `question_type` (categorical), `target_entity_iri` (gold answer IRI), `answer`, `text_answer`, `difficulty` (`easy`, `medium`, `hard`), `is_answerable` (boolean).
- Question types cover: task associations, dataset membership, publication year, implementation repositories, model variants, keyword retrieval — **TODO: verify full taxonomy count from dataset**.
- Difficulty levels are assigned per question — **TODO: document assignment criteria**.
- Questions span all three entity types (papers, datasets, models).

**Thesis-ready paragraph example:**

> The evaluation question set comprises 280 natural-language questions, each targeting a specific entity in the MLSea knowledge graph. Questions are annotated with a `question_type` field that categorises the retrieval intent — for example, questions asking which tasks a paper addresses, which datasets a method was evaluated on, or which model corresponds to a given implementation repository. Each question additionally carries a `difficulty` label (`easy`, `medium`, or `hard`) [TODO: document difficulty assignment criteria] and an `is_answerable` flag. Of the 280 questions, 265 are marked as answerable and are used in all metric computations; the remaining 15 are excluded from retrieval averages, as their gold target entities cannot be matched within the closed-world retrieval corpus.

**Figure/table:** Table 3.3 (question dataset field schema with types and example values).

**Source files:** `data/questions/ml_questions_dataset.json`

---

### 3.3.2 Gold Target Entity Design

**Goal:** Explain how ground-truth answers are represented and how retrieval success is defined.

**Key ideas:**
- Each question has a single gold target entity identified by its IRI (`target_entity_iri`).
- IRIs are normalised (stripped of trailing slashes, lowercased) before matching against candidate entity IDs.
- Closed-world assumption: the gold target entity is always present in the retrieval corpus (guaranteed by the gold-first curation step in §3.4.4).
- Single-gold-target design: each question has exactly one correct entity; there are no multiple-answer questions in this formulation.
- This design directly determines metric computation: rank of the gold IRI within the top-10 candidate list is the primary outcome variable.

**Figure/table:** Table 3.4 (gold target entity design — fields, normalisation rule, closed-world assumption).

---

### 3.3.3 Retrieval Task Formulation

**Goal:** Formally define the retrieval task as the system addresses it.

**Key ideas:**
- Input: a natural-language question string `q`.
- Output: a ranked list `L = [e_1, ..., e_k]` of entity identifiers, `k ≤ 10`.
- Success: `target_entity_iri ∈ L` (presence) and rank of `target_entity_iri` in `L` (for MRR/NDCG).
- `k = 10` is set in `src/retrieval/config.py` as `DEFAULT_TOP_K = 10`.
- The task is entity retrieval, not passage retrieval: each corpus unit corresponds to exactly one MLSea entity.
- Evaluation is segmented by entity type (paper, dataset, model), difficulty (easy, medium, hard), and question type.

**Figure/table:** Table 3.5 (retrieval task formulation — input, output, success criterion, segmentation dimensions).

**Source files:** `src/retrieval/config.py`; `data/questions/ml_questions_dataset.json`

---

### 3.4.1 Motivation for Pre-Retrieval Representations

**Goal:** Justify why raw RDF triples cannot serve as corpus units for dense retrieval and why entity-centric representation construction is necessary.

**Key ideas:**
- Problem 1 — Lexical incoherence: a single RDF triple such as `<pwc/scientificWork/bert> <mlso:hasTaskType> <pwc/task/question-answering>` is not a coherent natural-language unit. Its components are IRI strings, not human-readable text.
- Problem 2 — Two-level IRI indirection: object IRIs reference linked nodes (`pwc/task/question-answering`) whose human-readable label ("Question Answering") is stored on a different triple in the graph. A single-pass read cannot resolve this.
- Problem 3 — Distributional fragmentation: the information needed to answer "What tasks does the BERT paper address?" is distributed across many triples, none individually retrievable.
- Solution: aggregate all triples for one entity into a single entity-centric textual chunk; resolve linked-node IRIs to labels; the result is a semantically coherent, self-contained passage.

**Thesis-ready paragraph example:**

> Dense retrieval systems operate by comparing a query embedding against a corpus of embedded passages, where each passage is assumed to be a coherent, semantically self-contained unit of text. The MLSea knowledge graph, distributed as a collection of discrete RDF triples, does not satisfy this assumption. A triple of the form `<http://w3id.org/mlsea/pwc/scientificWork/bert> <mlso:hasTaskType> <http://w3id.org/mlsea/pwc/task/question-answering>` consists entirely of IRIs: the subject identifies the paper entity, the predicate specifies the relationship type, and the object is a linked task node whose human-readable label — "Question Answering" — is stored as a separate triple elsewhere in the graph. Even if individual triples were embedded, the resulting vectors would represent isolated predicate-object fragments rather than meaningful entity descriptions, and a question embedding for "What tasks does the BERT paper address?" would have no coherent passage to match against.

**Figure/table:** Figure 3.3 (before: raw RDF triple stream; after: entity-centric chunk).

**Source files:** `docs/post_retrieval/pre_retrieval_methodology.md`

---

### 3.4.2 RDF Extraction and Linked-Entity Label Resolution

**Goal:** Explain the two-pass streaming extraction pipeline that converts raw N-Triples into per-entity predicate-object collections with resolved labels.

**Key ideas:**
- Streaming line-by-line parse: necessary because the 6.4 GB file cannot be loaded into memory.
- Pass 1: for each line, if the subject IRI matches an entity prefix (paper, dataset, or model), record the predicate-object pair. Simultaneously collect all object IRIs that are referenced by entity subjects — these are linked nodes.
- Pass 2: for each linked-node IRI collected in Pass 1, retrieve its `rdfs:label`, `foaf:name`, `dcterms:title`, and `rdf:type` declarations. Store these in a node label cache keyed by IRI.
- Assembly: for each entity subject, combine its predicate-object pairs with the label cache to produce a structured entity record with human-readable values.
- Fallback: if no label predicate is found for a linked node, the IRI tail string is used as a fallback label (graceful degradation).
- Alternative rejected: loading into a triple store (GraphDB) and using SPARQL — originally prototyped but abandoned due to runtime dependency and scalability concerns.

**Thesis-ready paragraph example:**

> RDF extraction is implemented as a two-pass streaming scan of the N-Triples file. In the first pass, the parser reads each line and checks whether the subject IRI begins with one of the three recognised entity namespace prefixes. If so, the predicate-object pair is recorded under that subject, and the object IRI — if it refers to a linked node rather than a literal — is added to a set of unresolved references. The first pass thus produces, for each entity, a raw collection of predicate-value pairs and a set of linked-node IRIs that require label resolution. The second pass revisits the same file and, for each line where the subject IRI matches a previously collected linked-node IRI, records any `rdfs:label`, `foaf:name`, `dcterms:title`, or `schema:name` literal found. These labels are stored in a dictionary keyed by IRI, which is then used during entity record assembly to replace each linked-node IRI with its corresponding human-readable string.

**Figure/table:** Figure 3.4 (two-pass extraction schematic); Table 3.2 (predicate-to-field mapping).

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`, `src/pre_retrieval/shared/utils.py`

**Move to Chapter 4:** nothing — this is pure methodology.

---

### 3.4.3 Canonical Entity Record Construction

**Goal:** Describe how the raw per-entity predicate-object collections are normalised into a unified structured record that decouples extraction from chunk building.

**Key ideas:**
- Each predicate URI is mapped to a canonical field name: `dcterms:title` → `title`; `fabio:abstract` → `abstract`; `mlso:hasTaskType` → `tasks`; etc.
- Where multiple predicates can supply the same field (e.g., `dcterms:title`, `rdfs:label`, `foaf:name` all populate `title`), the first non-empty value in priority order is used.
- List-valued fields (tasks, datasets, methods, metrics, implementations) store resolved label strings from the linked-node label cache.
- The canonical record decouples extraction from chunk building: each of the 14 chunk builder scripts reads the same canonical record and selects the fields it needs.

**Paper record fields (verified from build_paper_records.py / v2 plan §4.3):**

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

**Figure/table:** Table 3.6 (canonical entity record fields — separate sub-table per entity type); Figure 3.5 (schematic of record construction).

**Source files:** `src/pre_retrieval/papers/raw/build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`

---

### 3.4.4 Corpus Curation and Subset Selection

**Goal:** Explain the construction of the 200,000-paper subset used for embedding and evaluation.

**Key ideas:**
- Full paper corpus (papers_master.jsonl) is large — **TODO: verify exact count**.
- Curation procedure (implemented in `build_curated_subset.py`):
  1. Collect all unique `target_entity_iri` values from the 280-question evaluation set.
  2. Include all gold-target papers unconditionally.
  3. Fill remaining capacity up to `max_papers = 200,000` (from `config/pre_retrieval_config.json`) with other papers from `papers_master.jsonl` in file order.
  4. Write to `data/intermediate/raw_papers/papers_subset_200k.jsonl`.
- Rationale for gold-first inclusion: guarantees that the closed-world evaluation is valid — every evaluation question has its correct answer present in the retrieval corpus.
- Rationale for 200k cap: computational constraint; embedding all papers across 6 strategies at 384 dimensions requires significant GPU time and storage.
- Datasets and models are not subject to the same cap (their corpora are smaller).

**Thesis-ready paragraph example:**

> To render the pre-retrieval experiments computationally feasible while preserving evaluation validity, the thesis works with a curated subset of 200,000 paper entities drawn from the full extracted paper corpus. The subset is constructed by a deterministic procedure: all papers that serve as gold-target entities for any evaluation question are included unconditionally, and the remaining positions are filled with additional papers from the full corpus in extraction order, up to the 200,000 limit. This gold-first inclusion guarantee ensures that the closed-world retrieval evaluation is well-formed: every evaluation question has its correct answer present in the retrieval corpus, and a failure to retrieve it reflects a genuine retrieval difficulty rather than a corpus coverage gap.

**Figure/table:** Table 3.7 (corpus construction summary — entity type, corpus size, gold target count, subset size).

**Source files:** `src/pre_retrieval/papers/raw/build_curated_subset.py`, `config/pre_retrieval_config.json`

---

### 3.4.5.1 Paper Representations (6 strategies)

**Goal:** Describe all six paper representation strategies, their field content, character limits, and design rationale.

**Key ideas (all limits verified from `config/pre_retrieval_config.json`):**

| Strategy | Fields Included | Max chars | Design rationale |
|---|---|---|---|
| `title_only` | Title | 512 | Minimum viable signal; pure title-matching baseline |
| `abstract_only` | Abstract | 1,600 | Tests semantic retrieval without the title anchor |
| `title_abstract` | Title + abstract | 1,800 (title ≤ 512, abstract ≤ 1,400) | Natural-language baseline combining the two most text-rich fields |
| `predicate_filtered` | Title + abstract (≤ 500) + tasks + datasets + methods + metrics (≤ 5 each) | 1,800 | Structured ML-domain fields; excludes authors and implementations which may introduce noise |
| `enriched_metadata` | Title (≤ 512) + abstract (≤ 900) + up to 5 tasks + 5 datasets + 5 methods + 5 metrics + 6 authors + 3 implementations | 2,200 | Maximum field coverage with per-field truncation |
| `one_hop` | Title (≤ 512) + abstract (≤ 700) + up to 12 linked entities grouped by inferred category | 2,200 | Graph-structure-centric; tests whether linked-entity grouping adds retrieval value |

**Thesis-ready paragraph example:**

> Six representation strategies are designed for paper entities, spanning from minimal to maximal information density. The simplest strategy, `title_only`, encodes only the paper title (up to 512 characters) and establishes the title-matching lower bound. The most information-dense strategy, `enriched_metadata`, concatenates the title, a truncated abstract (up to 900 characters), and up to five entries each for associated tasks, datasets, methods, and metrics, along with six authors and three implementation pointers, producing chunks of up to 2,200 characters. Between these extremes, `predicate_filtered` includes a shorter abstract (up to 500 characters) alongside structured ML-domain fields but deliberately excludes author lists and implementation URLs, which may introduce vocabulary noise for task-oriented questions. The `one_hop` strategy takes a graph-centric approach: rather than selecting fields by semantic category, it groups all resolved linked entities by their inferred RDF type (task, dataset, method, metric, implementation) and concatenates these groups as labelled sections.

**Figure/table:** Table 3.8 (chunk representation strategy matrix); Figure 3.6 (side-by-side chunk examples for `title_only`, `enriched_metadata`, `one_hop`).

**Source files:** `src/pre_retrieval/papers/chunking/` (6 builder scripts), `config/pre_retrieval_config.json`

**Move to Chapter 4:** NDCG values; which strategy is best; result interpretation.

---

### 3.4.5.2 Dataset Representations (4 strategies)

**Goal:** Describe the four dataset representation strategies, with emphasis on the annotation sparsity problem.

**Key ideas (limits from `config/pre_retrieval_config.json`):**

| Strategy | Fields Included | Max chars | Notes |
|---|---|---|---|
| `dataset_title_only` | Title/label | 512 | Minimal; reflects sparse annotation |
| `dataset_metadata` | Title + description + tasks + related papers (≤ 10 each) | 2,200 | Richer but many entities lack description |
| `dataset_predicate_filtered` | Title + description (≤ 500) + filtered predicates (≤ 5) | 1,800 | Selected subset of structured fields |
| `dataset_enriched_metadata` | Title + description (≤ 600) + related papers (≤ 6) + tasks (≤ 8) + implementations (≤ 4) + linked entities (≤ 6) | 2,400 | Maximum coverage |

**Key design observation:** Most dataset entities in MLSea carry only a title; adding more fields often adds noise rather than signal. This sparsity directly motivates the entity-type-specific representation comparison.

**Figure/table:** Table 3.8 (strategy matrix, dataset rows).

**Source files:** `src/pre_retrieval/datasets/chunking/`

---

### 3.4.5.3 Model Representations (4 strategies)

**Goal:** Describe the four model representation strategies, with emphasis on the predicate whitelist rationale.

**Key ideas (limits from `config/pre_retrieval_config.json`):**

| Strategy | Fields Included | Max chars | Notes |
|---|---|---|---|
| `model_title_only` | Label/title | 512 | Minimal; model names are often generic |
| `model_metadata` | Title + structured fields (≤ 10 each) | 2,200 | Broader but includes noisy linked entities |
| `model_predicate_filtered` | Title + curated predicate whitelist (≤ 5 each) | 1,800 | Whitelist selects task- and dataset-relevant predicates |
| `model_enriched_metadata` | Title + description (≤ 600) + linked entities (≤ 6) + structured fields (≤ 8) | 2,400 | Maximum coverage including graph links |

**Key design observation:** ML model names in MLSea are often generic (e.g., "CNN", "BERT variant"). The `model_predicate_filtered` strategy compensates by anchoring the representation in the model's functional role — its task associations and dataset links — via a curated predicate whitelist that excludes noisy or irrelevant linked-entity categories.

**Figure/table:** Table 3.8 (strategy matrix, model rows).

**Source files:** `src/pre_retrieval/models/chunking/`

---

### 3.4.6 Embedding Generation and Vector Indexing

**Goal:** Describe the embedding model, normalisation procedure, and ChromaDB vector indexing setup.

**Key ideas:**
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` — a 6-layer distilled SentenceTransformer producing 384-dimensional dense vectors [TODO: add MTEB citation and score].
- L2 normalisation applied at encoding time (`normalize_embeddings=True`): ensures cosine similarity equals dot-product similarity.
- Batch size: 64 per encoding call.
- Vector store: ChromaDB, persistent HNSW index, cosine metric.
- One ChromaDB collection per representation strategy — **TODO: verify total collection count (18 documented in v2 plan)**.
- Storage: `data/intermediate/chroma/` — **TODO: verify current store size (8.2 GB documented)**.
- Embedding query at retrieval time: questions are embedded with the same model and normalisation.

**Thesis-ready paragraph example:**

> All entity representations are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a distilled 6-layer SentenceTransformer model that produces 384-dimensional dense vectors [TODO: cite]. Embeddings are L2-normalised prior to storage, ensuring that cosine similarity is equivalent to dot-product similarity in the embedding space. Entity vectors are stored in ChromaDB, an embedded vector database that uses an HNSW index for approximate nearest-neighbour retrieval [TODO: cite HNSW]. A separate ChromaDB collection is created for each representation strategy, allowing per-strategy retrieval without cross-contamination between chunk types. At evaluation time, each question is embedded using the same model and normalisation procedure, and the resulting query vector is compared against the target collection to retrieve the top-10 most similar entity vectors by cosine distance.

**Figure/table:** Figure 3.7 (embedding and indexing workflow); Table 3.9 (embedding and indexing configuration).

**Source files:** `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py`, `config/pre_retrieval_config.json`

---

### 3.4.7 Pre-Retrieval Output

**Goal:** Describe what the pre-retrieval stage produces and how its output feeds the retrieval stage.

**Key ideas:**
- For each question × representation strategy combination, the pre-retrieval stage produces a ranked list of top-10 candidate entities with cosine similarity scores.
- Results are stored under `data/results/pre_retrieval_results/` in per-representation subdirectories.
- The best-performing representation per entity type — determined by NDCG evaluation (reported in Chapter 4) — is selected as the fixed input for the retrieval stage.
- Selected representations: `enriched_metadata` for papers, `dataset_title_only` for datasets, `model_predicate_filtered` for models.
- These selections are methodologically motivated here (the evaluation that justifies them is in Chapter 4).

**Source files:** `data/results/thesis_tables/best_per_entity.csv`, `data/results/pre_retrieval_results/`

**Move to Chapter 4:** NDCG values; full comparison tables; RQ1 answer.

---

### 3.5.1 Retrieval Objective and Design Rationale

**Goal:** Explain the transition from pre-retrieval to retrieval and justify the multi-method experimental design.

**Key ideas:**
- The retrieval stage takes the pre-retrieval top-10 output (for the best representation per entity type) as its input.
- Retrieval asks a distinct question from pre-retrieval: given the best possible semantic representation, do additional signals improve the candidate ranking?
- By fixing the representation and varying the retrieval method, the thesis isolates the contribution of symbolic and metadata-aware signals.
- Six methods are evaluated: one pure dense baseline, three hybrid symbolic-semantic methods, and two multi-representation fusion methods.
- Methods are organised in three families: Pure Semantic, Hybrid, and Optional Fusion.

**Thesis-ready paragraph example:**

> The retrieval stage receives the pre-retrieval top-10 candidate lists — produced using the best-performing representation per entity type — and applies one of six candidate generation strategies to produce the final ranked output. Whereas the pre-retrieval stage addresses the question of how to represent entities, the retrieval stage addresses the question of how to rank them: given that the most semantically relevant representations have been selected, can additional signals derived from the knowledge graph's structural metadata further improve the quality of the top-10 candidate list? The six methods evaluated in this stage are organised into three families. The first family contains the pure dense baseline, which passes the pre-retrieval ranking through unchanged. The second family contains three hybrid symbolic-semantic methods, each of which introduces a different form of structural signal: entity-type consistency filtering, one-hop graph-connectivity boosting, and question-type-to-predicate alignment. The third family contains two multi-representation fusion methods based on Reciprocal Rank Fusion.

**Figure/table:** Figure 3.8 (retrieval method workflow); Table 3.10 (retrieval method design matrix).

**Source files:** `docs/post_retrieval/retrieval_stage_plan.md`, `src/retrieval/config.py`

---

### 3.5.2 Dense Retrieval Baseline (`pure_semantic_dense`)

**Goal:** Describe the dense baseline as a passthrough of pre-retrieval results.

**Key ideas:**
- The dense baseline (`pure_semantic_dense`) passes the pre-retrieval top-10 results for the best representation per entity type directly to the output, without modification.
- It establishes the quality of pure semantic similarity ranking as the comparison reference for all hybrid methods.
- The baseline is strong because the pre-retrieval stage already selected the representation with the highest retrieval quality.
- Implemented in `src/retrieval/dense_baseline.py`.

**Figure/table:** Table 3.10 (row for `pure_semantic_dense`).

**Move to Chapter 4:** NDCG 0.7337; Hit@1 0.6717; all metric values.

---

### 3.5.3.1 Type-Filtered Retrieval (`hybrid_type_filtering`)

**Goal:** Describe type filtering as a control method that confirms collection purity.

**Key ideas:**
- Filters the dense baseline candidates to retain only those whose entity type matches the expected entity type (inferred from the `target_entity_iri` IRI prefix).
- Because pre-retrieval uses entity-type-specific ChromaDB collections, all candidates in the dense baseline already belong to the correct entity type — type filtering is a no-op by design.
- Implemented as a control: if `hybrid_type_filtering` produces different results from `pure_semantic_dense`, it would indicate a collection purity problem. If results are identical, collection purity is confirmed.
- Implemented in `src/retrieval/filtering.py:run_type_filtering`.

---

### 3.5.3.2 One-Hop Richness-Boosted Retrieval (`hybrid_type_onehop_filtering`)

**Goal:** Describe the graph-connectivity boosting method.

**Key ideas:**
- Applies type filtering first, then re-orders candidates by one-hop richness score.
- One-hop richness is computed as the count of non-empty metadata fields from the set {`tasks`, `datasets`, `methods`, `metrics`, `implementations`}, plus an additional bonus if the candidate's source text contains "Linked Entities" (indicating traversed one-hop links).
- Boost is question-type-agnostic: it tests whether better-connected graph entities are more likely to be answers regardless of question intent.
- Hypothesis: entities with richer one-hop connections are more informative and are more likely to be the correct answer.
- Implemented in `src/retrieval/filtering.py:run_hybrid_type_onehop_filtering`.

---

### 3.5.3.3 Predicate-Aware Retrieval (`hybrid_predicate_aware_filtering`)

**Goal:** Describe question-type-to-predicate alignment boosting.

**Key ideas:**
- Maps each question type to a specific metadata field and boosts candidates that have a non-empty value for that field.
- Boosted candidates are sorted before non-boosted candidates; within each group, the original dense ranking is preserved.
- Question type taxonomy (from `src/retrieval/filtering.py`):
  - Task types (`paper_to_tasks`, `paper_by_task_pair`, etc.) → boost candidates with non-empty `tasks`
  - Implementation types (`paper_to_implementation`) → boost candidates with non-empty `implementations`
  - Year types (`paper_to_publication_year`, `dataset_to_publication_year`) → boost candidates with non-null `publication_year`
  - Repository/family types (`repository_to_model`, `model_family_variant`, etc.) → boost candidates with "Linked Entities" in source text
  - Keyword types (`paper_to_keywords`) → boost candidates with non-empty `keywords`
- If no question type is matched (no recognised question type), candidates are returned in original order.
- Implemented in `src/retrieval/filtering.py:run_predicate_aware_filtering` via `_boost_by_predicate`.

---

### 3.5.4.1 Reciprocal Rank Fusion (`optional_rrf_fusion`)

**Goal:** Describe multi-representation rank fusion.

**Key ideas:**
- Fuses rankings from multiple representation strategies per entity type using Reciprocal Rank Fusion.
- **RRF formula:** `score(d) = Σ_{i} 1 / (k + rank_i(d))`, where k = 60 (from `src/retrieval/config.py: RRF_K = 60`), and the sum is over all representation rankings in which document `d` appears.
- Fusion groups (from `src/retrieval/config.py:RRF_FUSION_GROUPS`):
  - Papers: `enriched_metadata`, `predicate_filtered`, `one_hop` (3 representations)
  - Datasets: `dataset_title_only`, `dataset_enriched_metadata` (2 representations)
  - Models: `model_predicate_filtered`, `model_enriched_metadata` (2 representations)
- Rationale: different representations may rank different relevant entities highly; fusion broadens the recall base by combining these rankings.
- RRF is applied separately for each entity type.
- Implemented in `src/retrieval/rrf.py:run_rrf_fusion`.

**Thesis-ready paragraph example:**

> The `optional_rrf_fusion` method applies Reciprocal Rank Fusion (RRF) [TODO: cite Cormack et al. 2009] across multiple pre-retrieval representation rankings for the same entity type. For a candidate entity $d$ appearing in one or more representation-specific ranked lists, the RRF score is computed as:
>
> $$\text{score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$
>
> where $\text{rank}_i(d)$ is the rank of $d$ in the $i$-th representation's list and $k = 60$ is a smoothing constant that reduces the impact of top-ranked documents. Candidates are then sorted by RRF score in descending order, and the top 10 are returned. The fusion groups are: `enriched_metadata`, `predicate_filtered`, and `one_hop` for papers; `dataset_title_only` and `dataset_enriched_metadata` for datasets; `model_predicate_filtered` and `model_enriched_metadata` for models.

---

### 3.5.4.2 RRF with Symbolic Filtering (`optional_rrf_symbolic`)

**Goal:** Describe the combined fusion and symbolic boosting method.

**Key ideas:**
- Applies `optional_rrf_fusion` first to produce a fused candidate list.
- Then applies `hybrid_type_filtering` (type consistency check).
- Then applies `hybrid_predicate_aware_filtering` (question-type-to-predicate boost).
- Combines the breadth of RRF (broader recall) with the precision of predicate-aware boosting.
- Implemented in `src/retrieval/rrf.py:run_rrf_symbolic` as a composition of the three steps.

---

### 3.5.5 Top-k Candidate Generation

**Goal:** Explain the top-k design choice and the candidate output schema.

**Key ideas:**
- `DEFAULT_TOP_K = 10` (from `src/retrieval/config.py`).
- All six methods output the same candidate schema.
- k=10 defines the recall ceiling: if the gold entity is not in the top-10, no downstream re-ranker can recover it.
- Candidate fields include: entity IRI (normalised), entity type, rank within the list, method score, source text (the chunk used for embedding), and metadata fields (title, tasks, publication_year, etc.).
- The `result_schema.py` module standardises the output format across all methods.

**Figure/table:** Table 3.11 (top-k candidate output schema).

**Source files:** `src/retrieval/config.py`, `src/retrieval/pipeline.py`, `src/retrieval/result_schema.py`

---

### 3.5.6 Retrieval Output

**Goal:** Describe what the retrieval stage delivers and connect it to the post-retrieval stage.

**Key ideas:**
- Each retrieval method produces, per question: a ranked list of ≤10 candidate entity records and aggregated metrics.
- Results are stored under `data/results/retrieval/{method_name}/` as `results.json` (per-question) and `metrics.json` (aggregated).
- The ranked candidate list is the primary handoff artefact: it constitutes the evidence base that the downstream post-retrieval stage will process.
- Hit@10 defines the maximum recoverability of downstream stages: a gold entity absent from the top-10 cannot be recovered by any subsequent re-ranker or answer generator.
- The post-retrieval stage receives this candidate set and proceeds with context assembly, potential re-ranking, and answer generation. The methodology of the post-retrieval stage is introduced as the third component of the complete KG-RAG pipeline.

**Thesis-ready paragraph example:**

> The retrieval stage produces, for each evaluation question, a ranked list of at most ten candidate entity records selected by the applied retrieval method. This ranked list constitutes the retrieval output of the pipeline: it is a structured, ranked set of evidence entities that the subsequent post-retrieval stage will use to assemble a context for answer generation. The composition of this candidate set determines the maximum performance achievable by any downstream processing: if the gold target entity does not appear within the top-ten candidates, it cannot be recovered regardless of the re-ranking or generation strategy applied in the post-retrieval stage. The evaluation of retrieval quality — the metrics, the method comparison, and the answers to RQ2 — is presented in Chapter 4. The methodology developed in this chapter provides the evidence base for downstream post-retrieval processing.

**Figure/table:** Figure 3.9 (top-k retrieval output and downstream handoff).

**Source files:** `src/retrieval/result_schema.py`, `data/results/retrieval/`

---

## 4. Chapter 3 Figures and Tables Plan

### 4.1 Figures

---

**Figure 3.1 — Full KG-RAG Pipeline Overview**

| Field | Content |
|---|---|
| Number | Figure 3.1 |
| Title | Full KG-RAG Pipeline Overview |
| Purpose | Give the reader a visual map of the three-stage system before details are presented |
| Where to insert | §3.1 Methodological Overview, immediately after opening paragraph |
| What it should show | Three horizontal boxes: Pre-Retrieval (RDF → chunk → embed → index), Retrieval (question → embed → rank → candidates), Post-Retrieval (candidates → context → generate → answer); arrows connecting stages; RQ labels per stage |
| Source data | Conceptual — no existing data file; must be drawn |
| Exists? | No — must be created |
| Recommended caption | "The three-stage KG-RAG pipeline developed in this thesis. The pre-retrieval stage converts MLSea RDF entities into dense-retrievable textual representations. The retrieval stage ranks candidate entities by embedding-based similarity and symbolic-semantic hybrid methods. The post-retrieval stage assembles the retrieved candidate set into a context for answer generation." |
| LaTeX label | `\label{fig:pipeline_overview}` |

---

**Figure 3.2 — MLSea RDF Entity Structure Example**

| Field | Content |
|---|---|
| Number | Figure 3.2 |
| Title | MLSea RDF Entity Structure: Example N-Triples |
| Purpose | Show the reader what raw RDF looks like before any processing |
| Where to insert | §3.2.1 The MLSea Knowledge Graph |
| What it should show | 5–8 example N-Triples lines for one paper entity, with subject/predicate/object columns highlighted; one linked-node triple showing that the object IRI needs label resolution |
| Source data | `data/raw/pwc_1.nt` — sample lines for any gold-target paper entity |
| Exists? | No — must be drawn/formatted as a code listing or table |
| Recommended caption | "Example N-Triples from the MLSea knowledge graph for a paper entity. The subject IRI identifies the paper; predicates encode relationships such as title, task association, and authorship; object IRIs reference linked nodes whose human-readable labels must be resolved in a second extraction pass." |
| LaTeX label | `\label{fig:rdf_example}` |

---

**Figure 3.3 — RDF Triples to Entity-Centric Chunk**

| Field | Content |
|---|---|
| Number | Figure 3.3 |
| Title | From Distributed RDF Triples to Entity-Centric Textual Chunk |
| Purpose | Visually justify the need for pre-retrieval representation construction |
| Where to insert | §3.4.1 Motivation for Pre-Retrieval Representations |
| What it should show | Left panel: 6–8 disconnected RDF triples for one paper entity (with IRI objects); Right panel: the resulting entity-centric chunk with all fields resolved to human-readable text |
| Source data | Conceptual; can use real data from JSONL files for the resolved chunk |
| Exists? | No — must be created |
| Recommended caption | "A paper entity represented as raw RDF triples (left) versus its corresponding entity-centric textual chunk (right). Raw triples are lexically incoherent and structurally fragmented; the entity-centric chunk resolves linked-node IRIs to human-readable labels and aggregates all predicate values into a single dense-retrievable passage." |
| LaTeX label | `\label{fig:rdf_to_chunk}` |

---

**Figure 3.4 — Two-Pass RDF Extraction and Linked-Entity Label Resolution**

| Field | Content |
|---|---|
| Number | Figure 3.4 |
| Title | Two-Pass Streaming RDF Extraction and Label Resolution |
| Purpose | Show the two-pass extraction algorithm as a data flow diagram |
| Where to insert | §3.4.2 RDF Extraction and Linked-Entity Label Resolution |
| What it should show | Pass 1: N-Triples stream → entity predicate-object collection + linked IRI set; Pass 2: N-Triples stream → linked IRI → label cache; Assembly: entity record per entity |
| Source data | `src/pre_retrieval/papers/raw/build_paper_records.py` |
| Exists? | No — must be created |
| Recommended caption | "The two-pass streaming extraction pipeline. Pass 1 collects predicate-object pairs for each entity subject and records linked-node IRIs; Pass 2 resolves each linked-node IRI to a human-readable label using `rdfs:label` or equivalent predicates. The two passes are combined during entity record assembly." |
| LaTeX label | `\label{fig:two_pass_extraction}` |

---

**Figure 3.5 — Canonical Entity Record Construction**

| Field | Content |
|---|---|
| Number | Figure 3.5 |
| Title | Canonical Entity Record Construction |
| Purpose | Show how raw predicate-object pairs and resolved labels are mapped to a structured record |
| Where to insert | §3.4.3 Canonical Entity Record Construction |
| What it should show | Input: predicate-object pairs + label cache → Output: canonical record with named fields (title, abstract, tasks, datasets, etc.) |
| Source data | `src/pre_retrieval/papers/raw/build_paper_records.py`; field schema from Table 3.6 |
| Exists? | No — must be created |
| Recommended caption | "Canonical entity record construction for a paper entity. Raw predicate-object pairs are normalised to canonical field names, and linked-node IRIs are replaced with their resolved label strings. The resulting canonical record is the unified input for all chunk builder scripts." |
| LaTeX label | `\label{fig:canonical_record}` |

---

**Figure 3.6 — Example Chunk Representations**

| Field | Content |
|---|---|
| Number | Figure 3.6 |
| Title | Example Entity-Centric Chunk Representations for the Same Paper |
| Purpose | Illustrate the contrast between representation strategies on a concrete example |
| Where to insert | §3.4.5 Entity-Centric Chunk Construction |
| What it should show | Three side-by-side text boxes for the same paper: `title_only`, `enriched_metadata`, `one_hop` — showing how field selection changes chunk content and length |
| Source data | JSONL files under `data/intermediate/representations/papers/` for a gold-target paper |
| Exists? | No — must be extracted from JSONL files |
| Recommended caption | "Entity-centric chunk representations for the same paper entity under three strategies. `title_only` encodes only the title (up to 512 characters). `enriched_metadata` adds abstract, task associations, dataset links, methods, metrics, and authors (up to 2,200 characters). `one_hop` groups all resolved linked entities by inferred RDF type (up to 2,200 characters)." |
| LaTeX label | `\label{fig:chunk_examples}` |

---

**Figure 3.7 — Embedding and Vector Indexing Workflow**

| Field | Content |
|---|---|
| Number | Figure 3.7 |
| Title | Embedding Generation and ChromaDB Vector Indexing Workflow |
| Purpose | Show the full embedding pipeline from chunk text to stored vector |
| Where to insert | §3.4.6 Embedding Generation and Vector Indexing |
| What it should show | Chunk text → `all-MiniLM-L6-v2` encoder → 384-dim L2-normalised vector → ChromaDB HNSW collection (one per strategy) |
| Source data | `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py` |
| Exists? | No — must be created |
| Recommended caption | "Embedding and vector indexing workflow. Each entity chunk is encoded by `sentence-transformers/all-MiniLM-L6-v2` to produce a 384-dimensional L2-normalised vector, which is stored in a dedicated ChromaDB collection using an HNSW index with cosine distance." |
| LaTeX label | `\label{fig:embedding_workflow}` |

---

**Figure 3.8 — Retrieval Method Workflow**

| Field | Content |
|---|---|
| Number | Figure 3.8 |
| Title | Retrieval Method Workflow: From Question to Ranked Candidates |
| Purpose | Show how the six retrieval methods relate to each other and to the pre-retrieval output |
| Where to insert | §3.5.1 Retrieval Objective and Design Rationale |
| What it should show | Pre-retrieval top-10 (best representation) → six branches: dense passthrough; type filter; type + one-hop; predicate-aware; RRF fusion; RRF + symbolic → final ranked candidate list |
| Source data | `src/retrieval/pipeline.py`, `config.py` |
| Exists? | No — must be created |
| Recommended caption | "The six retrieval methods evaluated in the retrieval stage. All methods take the pre-retrieval top-10 candidate lists as input. The pure dense baseline passes them through unchanged. The hybrid and fusion methods apply additional symbolic or structural signals to re-order or expand the candidate set." |
| LaTeX label | `\label{fig:retrieval_workflow}` |

---

**Figure 3.9 — Top-k Retrieval Output and Downstream Use**

| Field | Content |
|---|---|
| Number | Figure 3.9 |
| Title | Top-k Retrieval Output and Handoff to Post-Retrieval |
| Purpose | Close the chapter by showing the retrieval output format and its role as the post-retrieval input |
| Where to insert | §3.5.6 Retrieval Output |
| What it should show | Ranked list of top-10 entity records (entity IRI, type, rank, score, source text) → arrow → post-retrieval stage (context assembly, re-ranking, generation) |
| Source data | `src/retrieval/result_schema.py` |
| Exists? | No — must be created |
| Recommended caption | "The top-10 retrieval output constitutes the evidence base for the downstream post-retrieval stage. Each candidate record carries the entity IRI, entity type, rank within the list, method score, and the source text chunk used for embedding. The post-retrieval stage receives this ranked list and proceeds with context assembly and answer generation." |
| LaTeX label | `\label{fig:retrieval_output}` |

---

### 4.2 Tables

---

**Table 3.1 — MLSea Entity Types and Role in the Pipeline**

| Field | Content |
|---|---|
| Number | Table 3.1 |
| Title | MLSea Entity Types and Their Role in the KG-RAG Pipeline |
| Purpose | Summarise the three entity types at a glance |
| Where to insert | §3.2.2 Entity Types and RDF Structure |
| What it should show | Columns: Entity type, IRI prefix, Typical annotation density, Primary metadata fields, Role in thesis |
| Source data | `CLAUDE.md`, `docs/post_retrieval/thesis_overview.md`, `pre_retrieval_methodology.md` |
| Exists? | No — must be written |
| Recommended caption | "The three entity types in the MLSea knowledge graph and their role in the KG-RAG pipeline. Annotation density varies substantially across entity types and motivates the entity-type-specific representation strategies developed in Section 3.4.5." |
| LaTeX label | `\label{tab:entity_types}` |

---

**Table 3.2 — RDF Predicate-to-Field Mapping**

| Field | Content |
|---|---|
| Number | Table 3.2 |
| Title | RDF Predicate-to-Field Mapping for Papers, Datasets, and Models |
| Purpose | Document which RDF predicates map to which canonical record fields |
| Where to insert | §3.4.2 / §3.4.3 |
| What it should show | Columns: Canonical field, Source predicate(s), Entity types, Notes |
| Source data | `src/pre_retrieval/papers/raw/build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` |
| Exists? | No — must be extracted from source code |
| Recommended caption | "RDF predicate-to-canonical-field mapping used during entity record construction. Where multiple predicates can supply the same field, the first non-empty value in priority order is used." |
| LaTeX label | `\label{tab:predicate_mapping}` |

---

**Table 3.3 — Question Dataset Schema**

| Field | Content |
|---|---|
| Number | Table 3.3 |
| Title | Question Dataset Field Schema |
| Purpose | Document the structure of the evaluation question set |
| Where to insert | §3.3.1 Question Dataset |
| What it should show | Columns: Field name, Type, Description, Example value |
| Source data | `data/questions/ml_questions_dataset.json` |
| Exists? | No — must be formatted from inspection of JSON file |
| Recommended caption | "Schema of the evaluation question dataset. Each of the 280 questions carries a natural-language question string, a categorical question type, a single gold target entity IRI, a difficulty label, and an answerability flag." |
| LaTeX label | `\label{tab:question_schema}` |

---

**Table 3.4 — Gold Target Entity Design**

| Field | Content |
|---|---|
| Number | Table 3.4 |
| Title | Gold Target Entity Design |
| Purpose | Explain how ground-truth answers are defined and matched |
| Where to insert | §3.3.2 Gold Target Entity Design |
| What it should show | Columns: Design element, Definition, Rationale — rows: single gold IRI, IRI normalisation, closed-world assumption, one-correct-entity-per-question |
| Source data | `data/questions/ml_questions_dataset.json`, evaluation setup |
| Exists? | No — must be written |
| Recommended caption | "Gold target entity design. Each evaluation question has a single gold target entity IRI. IRIs are normalised before matching to avoid format mismatches. The closed-world assumption is enforced by the gold-first corpus curation procedure (Section 3.4.4)." |
| LaTeX label | `\label{tab:gold_target}` |

---

**Table 3.5 — Retrieval Task Formulation**

| Field | Content |
|---|---|
| Number | Table 3.5 |
| Title | Retrieval Task Formulation |
| Purpose | Formally specify the retrieval task |
| Where to insert | §3.3.3 Retrieval Task Formulation |
| What it should show | Rows: Input, Output, Success criterion, Default k, Segmentation dimensions |
| Source data | `src/retrieval/config.py` (DEFAULT_TOP_K=10), `data/questions/ml_questions_dataset.json` |
| Exists? | No — must be written |
| Recommended caption | "Formal retrieval task specification. The system receives a natural-language question and must produce a ranked list of at most ten entity identifiers. Success is measured by the rank of the gold target entity within this list." |
| LaTeX label | `\label{tab:retrieval_task}` |

---

**Table 3.6 — Canonical Entity Record Fields**

| Field | Content |
|---|---|
| Number | Table 3.6 |
| Title | Canonical Entity Record Fields |
| Purpose | Document the unified structured record format for each entity type |
| Where to insert | §3.4.3 Canonical Entity Record Construction |
| What it should show | Separate sub-tables for papers, datasets, models — columns: Field name, Source predicate(s), Type, Notes |
| Source data | `src/pre_retrieval/papers/raw/build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` |
| Exists? | No — must be extracted from source code |
| Recommended caption | "Canonical entity record fields for papers, datasets, and models. All chunk builder scripts read from this unified record format, ensuring that field selection in each strategy is well-defined and consistent." |
| LaTeX label | `\label{tab:canonical_record}` |

---

**Table 3.7 — Corpus Construction Summary**

| Field | Content |
|---|---|
| Number | Table 3.7 |
| Title | Corpus Construction Summary |
| Purpose | Document the entity counts and curation logic |
| Where to insert | §3.4.4 Corpus Curation and Subset Selection |
| What it should show | Columns: Entity type, Full corpus size, Gold targets in evaluation set, Subset size, Curation rule |
| Source data | `src/pre_retrieval/papers/raw/build_curated_subset.py`, `config/pre_retrieval_config.json`, `data/questions/ml_questions_dataset.json` |
| Exists? | No — paper full corpus size is TODO; other values available |
| Recommended caption | "Entity corpus construction summary. The paper subset is limited to 200,000 entities with gold-first inclusion. Dataset and model corpora are used in full." |
| LaTeX label | `\label{tab:corpus_summary}` |

---

**Table 3.8 — Chunk Representation Strategy Matrix**

| Field | Content |
|---|---|
| Number | Table 3.8 |
| Title | Entity-Centric Chunk Representation Strategy Matrix |
| Purpose | Summarise all 14 strategies with fields and character limits |
| Where to insert | §3.4.5 Entity-Centric Chunk Construction |
| What it should show | Columns: Entity type, Strategy name, Fields included, Max characters, Key design rationale — 14 rows (6 paper + 4 dataset + 4 model) |
| Source data | `config/pre_retrieval_config.json`; chunking builder scripts |
| Exists? | No — must be formatted from config and scripts |
| Recommended caption | "Chunk representation strategy matrix. All character limits are enforced at chunk construction time. Strategies span from minimal field coverage (title only, 512 chars) to maximum field coverage (enriched metadata, up to 2,400 chars)." |
| LaTeX label | `\label{tab:representation_matrix}` |

---

**Table 3.9 — Embedding and Indexing Configuration**

| Field | Content |
|---|---|
| Number | Table 3.9 |
| Title | Embedding and Vector Indexing Configuration |
| Purpose | Document the full embedding and ChromaDB setup |
| Where to insert | §3.4.6 Embedding Generation and Vector Indexing |
| What it should show | Rows: Model, Vector dimension, Normalisation, Similarity metric, Vector store, Index type, Collections, Storage path, Batch size |
| Source data | `config/pre_retrieval_config.json`, `src/pre_retrieval/shared/embedder.py`, `vector_store.py` |
| Exists? | No — must be written |
| Recommended caption | "Embedding and vector indexing configuration. One ChromaDB collection is created per representation strategy. All collections use cosine distance with L2-normalised vectors." |
| LaTeX label | `\label{tab:indexing_config}` |

---

**Table 3.10 — Retrieval Method Design Matrix**

| Field | Content |
|---|---|
| Number | Table 3.10 |
| Title | Retrieval Method Design Matrix |
| Purpose | Summarise all six retrieval methods with their family, input, and symbolic signal |
| Where to insert | §3.5.1 Retrieval Objective and Design Rationale |
| What it should show | Columns: Method name, Family, Input, Symbolic signal applied, Design hypothesis, Source file |
| Source data | `src/retrieval/config.py`, `docs/post_retrieval/retrieval_stage_plan.md` |
| Exists? | No — must be written |
| Recommended caption | "Retrieval method design matrix. All methods take the pre-retrieval top-10 candidate list as input. Methods in the Hybrid and Optional Fusion families apply additional symbolic signals derived from the MLSea knowledge graph structure." |
| LaTeX label | `\label{tab:retrieval_methods}` |

---

**Table 3.11 — Top-k Candidate Output Schema**

| Field | Content |
|---|---|
| Number | Table 3.11 |
| Title | Top-k Candidate Output Schema |
| Purpose | Document the structured format of each retrieval output record |
| Where to insert | §3.5.5 Top-k Candidate Generation |
| What it should show | Columns: Field name, Type, Description — rows for entity IRI, entity type, rank, method score, source text, metadata fields |
| Source data | `src/retrieval/result_schema.py` |
| Exists? | No — must be extracted from result_schema.py |
| Recommended caption | "Top-k candidate output schema. All six retrieval methods produce candidates conforming to this schema, enabling consistent downstream processing in the post-retrieval stage." |
| LaTeX label | `\label{tab:candidate_schema}` |

---

## 5. Algorithm Boxes for Chapter 3

Pseudocode is provided for methodology description only. Metric computation algorithms belong in Chapter 4.

---

**Algorithm 1 — Two-Pass RDF Extraction**

```
Algorithm 1: Two-Pass RDF Extraction
Input:  N-Triples file F, entity IRI prefixes P = {papers_prefix, dataset_prefix, model_prefix}
Output: Dict entity_records[entity_IRI] → list of (predicate, object_or_label) pairs,
        Dict label_cache[linked_IRI] → (label_string, category)

Pass 1:
  entity_predicates ← {}  // entity_IRI → list[(predicate, object)]
  linked_iris ← {}        // IRI → True if referenced as object by an entity subject

  for each line (s, p, o) in F:
    if s starts with any prefix in P:
      entity_predicates[s].append((p, o))
      if o is IRI (not literal):
        linked_iris.add(o)

Pass 2:
  label_cache ← {}
  label_predicates ← {rdfs:label, foaf:name, dcterms:title, schema:name}
  type_predicate ← rdf:type

  for each line (s, p, o) in F:
    if s ∈ linked_iris:
      if p ∈ label_predicates and label_cache[s].label is None:
        label_cache[s].label ← o
      if p = type_predicate:
        label_cache[s].types.add(o)

Assembly:
  for each entity_IRI in entity_predicates:
    record ← {}
    for (predicate, object) in entity_predicates[entity_IRI]:
      canonical_field ← PREDICATE_TO_FIELD_MAP[predicate]
      if object ∈ label_cache:
        value ← label_cache[object].label  // resolved label
      else:
        value ← object  // literal or IRI fallback
      record[canonical_field].append(value)
    entity_records[entity_IRI] ← record

Source: src/pre_retrieval/papers/raw/build_paper_records.py,
        src/pre_retrieval/shared/utils.py
```

---

**Algorithm 2 — Linked-Entity Label Resolution**

```
Algorithm 2: Linked-Entity Label Resolution
Input:  Set linked_iris, N-Triples file F
Output: Dict label_cache[IRI] → {label: str, types: set, category: str}

label_predicates ← {rdfs:label, foaf:name, dcterms:title, schema:name}
category_map ← {task IRI prefix → "task", dataset IRI prefix → "dataset", ...}

label_cache ← {iri: {label: None, types: set()} for iri in linked_iris}

for each line (s, p, o) in F:
  if s ∈ label_cache:
    if p ∈ label_predicates and label_cache[s].label is None:
      label_cache[s].label ← o (strip language tag if present)
    if p = rdf:type:
      label_cache[s].types.add(o)

for iri in label_cache:
  if label_cache[iri].label is None:
    label_cache[iri].label ← iri.split("/")[-1]  // IRI tail as fallback
  label_cache[iri].category ← infer_category(label_cache[iri].types, iri)

Source: src/pre_retrieval/papers/raw/build_paper_records.py,
        src/pre_retrieval/shared/utils.py
```

---

**Algorithm 3 — Canonical Entity Record Construction**

```
Algorithm 3: Canonical Entity Record Construction
Input:  entity_predicates[entity_IRI], label_cache
Output: canonical_record: dict with typed fields

PREDICATE_TO_FIELD ← {
  dcterms:title → title (priority 1),
  rdfs:label    → title (priority 2),
  foaf:name     → title (priority 3),
  fabio:abstract → abstract,
  dcterms:issued → publication_year,
  dcterms:creator → authors (list),
  dcat:keyword  → keywords (list),
  mlso:hasTaskType → tasks (list, resolve via label_cache),
  ...
}

for entity_IRI, raw_pairs in entity_predicates:
  record ← defaultdict(list)
  for (predicate, object) in raw_pairs:
    if predicate ∈ PREDICATE_TO_FIELD:
      field ← PREDICATE_TO_FIELD[predicate]
      if object ∈ label_cache:
        value ← label_cache[object].label
        if value not already in record[field]:
          record[field].append(value)
      else:
        record[field].append(object)  // literal value
  deduplicate and normalise all list fields
  yield record

Source: src/pre_retrieval/papers/raw/build_paper_records.py,
        src/pre_retrieval/datasets/raw/build_dataset_records.py,
        src/pre_retrieval/models/raw/build_model_records.py
```

---

**Algorithm 4 — Entity-Centric Chunk Construction**

```
Algorithm 4: Entity-Centric Chunk Construction
Input:  canonical_record, strategy_config (field limits, max_chars)
Output: chunk_text: str (UTF-8, within max_chars limit)

// Example for enriched_metadata strategy
function build_enriched_metadata_chunk(record, config):
  parts ← []
  parts.append("Title: " + truncate(record.title, config.title_max_chars))
  if record.abstract:
    parts.append("Abstract: " + truncate(record.abstract, config.abstract_max_chars))
  if record.tasks:
    parts.append("Tasks: " + join(record.tasks[:config.list_item_limit], "; "))
  if record.datasets:
    parts.append("Datasets: " + join(record.datasets[:config.list_item_limit], "; "))
  if record.methods:
    parts.append("Methods: " + join(record.methods[:config.list_item_limit], "; "))
  if record.metrics:
    parts.append("Metrics: " + join(record.metrics[:config.list_item_limit], "; "))
  if record.authors:
    parts.append("Authors: " + join(record.authors[:config.author_limit], ", "))
  if record.implementations:
    parts.append("Code: " + join(record.implementations[:config.implementation_limit], "; "))
  chunk ← join(parts, " | ")
  return truncate(chunk, config.max_chars)

// Other strategies (title_only, abstract_only, predicate_filtered, one_hop) follow analogous
// field-selection logic as defined in config/pre_retrieval_config.json

Source: src/pre_retrieval/papers/chunking/build_enriched_paper_chunks.py (and equivalents)
        config/pre_retrieval_config.json
```

---

**Algorithm 5 — Embedding and ChromaDB Indexing**

```
Algorithm 5: Embedding and ChromaDB Indexing
Input:  chunk_records (entity_id, chunk_text, metadata), collection_name, config
Output: populated ChromaDB collection

model ← SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
collection ← ChromaDB.get_or_create_collection(collection_name, metric="cosine")

for batch in chunk_records (batch_size=64):
  texts ← [r.chunk_text for r in batch]
  embeddings ← model.encode(texts, normalize_embeddings=True)  // 384-dim L2-normalised
  collection.add(
    ids=[r.entity_id for r in batch],
    embeddings=embeddings,
    documents=texts,
    metadatas=[r.metadata for r in batch]
  )

Source: src/pre_retrieval/shared/embedder.py,
        src/pre_retrieval/shared/embed_and_store.py,
        src/pre_retrieval/shared/vector_store.py
```

---

**Algorithm 6 — Dense Retrieval Baseline**

```
Algorithm 6: Dense Retrieval Baseline (pure_semantic_dense)
Input:  questions (list), top_k=10, best_representation per entity_type
Output: per-question ranked candidate lists

for question in questions:
  entity_type ← infer_entity_type(question.target_entity_iri)
  representation ← BEST_REPRESENTATION[entity_type]
  // BEST_REPRESENTATION: paper→enriched_metadata, dataset→dataset_title_only,
  //                       model→model_predicate_filtered
  
  pre_retrieval_result ← load_top10(entity_type, representation, question.id)
  candidates ← pre_retrieval_result.candidates  // already ranked by cosine similarity
  yield make_question_result(question, method="pure_semantic_dense", candidates=candidates)

Source: src/retrieval/dense_baseline.py,
        src/retrieval/config.py (BEST_REPRESENTATION)
```

---

**Algorithm 7 — Type-Filtered Retrieval**

```
Algorithm 7: Type-Filtered Retrieval (hybrid_type_filtering)
Input:  dense_results (from Algorithm 6), top_k=10
Output: per-question candidate lists filtered by expected entity type

for result in dense_results:
  expected_type ← infer_entity_type(result.target_entity_iri)
  candidates ← result.candidates
  filtered ← [c for c in candidates if c.entity_type == expected_type]
  if not filtered:
    filtered ← candidates  // revert to unfiltered if no typed candidates found (warning logged)
  yield make_question_result(..., method="hybrid_type_filtering",
                             candidates=renumber(filtered[:top_k]))

Source: src/retrieval/filtering.py:run_type_filtering
```

---

**Algorithm 8 — One-Hop Richness-Boosted Retrieval**

```
Algorithm 8: One-Hop Richness-Boosted Retrieval (hybrid_type_onehop_filtering)
Input:  dense_results, top_k=10
Output: per-question candidate lists re-ranked by one-hop richness

ONEHOP_FIELDS ← {tasks, datasets, methods, metrics, implementations}

function onehop_richness(candidate):
  score ← count of non-empty fields in ONEHOP_FIELDS from candidate.metadata
  if "Linked Entities" in candidate.metadata.source_text:
    score += 2
  return score

for result in dense_results:
  expected_type ← infer_entity_type(result.target_entity_iri)
  type_filtered ← [c for c in result.candidates if c.entity_type == expected_type]
  reranked ← sorted(type_filtered, key=onehop_richness, reverse=True)
  yield make_question_result(..., method="hybrid_type_onehop_filtering",
                             candidates=renumber(reranked[:top_k]))

Source: src/retrieval/filtering.py:run_hybrid_type_onehop_filtering,
        src/retrieval/filtering.py:_onehop_richness
```

---

**Algorithm 9 — Predicate-Aware Retrieval**

```
Algorithm 9: Predicate-Aware Retrieval (hybrid_predicate_aware_filtering)
Input:  dense_results, top_k=10
Output: per-question candidate lists with predicate-matched candidates promoted

QUESTION_TYPE_TO_PREDICATE ← {
  {paper_to_tasks, paper_by_task_pair, ...} → check non-empty tasks,
  {paper_to_implementation}                 → check non-empty implementations,
  {paper_to_publication_year, ...}          → check non-null publication_year,
  {repository_to_model, model_family_...}  → check "Linked Entities" in source_text,
  {paper_to_keywords}                       → check non-empty keywords
}

function boost_by_predicate(candidates, question_type):
  predicate_check ← QUESTION_TYPE_TO_PREDICATE.get(question_type, None)
  if predicate_check is None:
    return candidates, boosted=False
  boosted ← [c for c in candidates if predicate_check(c)]
  rest    ← [c for c in candidates if not predicate_check(c)]
  return boosted + rest, boosted=len(boosted)>0

for result in dense_results:
  reordered, _ ← boost_by_predicate(result.candidates, result.question_type)
  yield make_question_result(..., method="hybrid_predicate_aware_filtering",
                             candidates=renumber(reordered[:top_k]))

Source: src/retrieval/filtering.py:run_predicate_aware_filtering,
        src/retrieval/filtering.py:_boost_by_predicate
```

---

**Algorithm 10 — Reciprocal Rank Fusion**

```
Algorithm 10: Reciprocal Rank Fusion (optional_rrf_fusion)
Input:  pre_retrieval_indexes {entity_type → {representation → {question_id → top10_result}}},
        questions, top_k=10, k=60
Output: per-question RRF-fused candidate lists

RRF_FUSION_GROUPS ← {
  paper: [enriched_metadata, predicate_filtered, one_hop],
  dataset: [dataset_title_only, dataset_enriched_metadata],
  model: [model_predicate_filtered, model_enriched_metadata]
}

function rrf_score(rank, k=60):
  return 1.0 / (k + rank)

for question in questions:
  entity_type ← infer_entity_type(question.target_entity_iri)
  representations ← RRF_FUSION_GROUPS[entity_type]
  scores ← defaultdict(float)  // entity_id → cumulative RRF score

  for representation in representations:
    top10 ← pre_retrieval_indexes[entity_type][representation][question.id]
    for candidate in top10.candidates:
      scores[candidate.entity_id] += rrf_score(candidate.rank, k)

  sorted_ids ← sort by scores[id] descending
  fused_candidates ← top_k entities from sorted_ids with scores
  yield make_question_result(..., method="optional_rrf_fusion", candidates=fused_candidates)

Source: src/retrieval/rrf.py:run_rrf_fusion,
        src/retrieval/rrf.py:_rrf_score,
        src/retrieval/config.py (RRF_K=60, RRF_FUSION_GROUPS)
```

---

**Algorithm 11 — RRF with Symbolic Filtering**

```
Algorithm 11: RRF with Symbolic Filtering (optional_rrf_symbolic)
Input:  pre_retrieval_indexes, questions, top_k=10, k=60
Output: per-question candidate lists combining RRF fusion with symbolic signals

Step 1: rrf_results ← run_rrf_fusion(questions, top_k, k)  // Algorithm 10
Step 2: type_results ← run_type_filtering(rrf_results, top_k)  // Algorithm 7
Step 3: final_results ← run_predicate_aware_filtering(type_results, top_k)  // Algorithm 9

Relabel method_name ← "optional_rrf_symbolic" throughout
yield final_results

Source: src/retrieval/rrf.py:run_rrf_symbolic
```

---

## 6. Chapter 3 Source-to-Section Traceability

| Chapter 3 Section | Main Methodology Claim | Source Code Files | Config Files | Documentation Files | Result Files (if needed for selected inputs) | Citation Needed? | TODO |
|---|---|---|---|---|---|---|---|
| 3.1 | Three-stage KG-RAG pipeline: pre-retrieval → retrieval → post-retrieval | — | — | `thesis_overview.md`, `CLAUDE.md` | — | RAG [TODO] | None |
| 3.2.1 | MLSea is a 26,606,202-triple N-Triples RDF file, 6.4 GB | `build_paper_records.py` | — | `CLAUDE.md` | — | MLSea [TODO] | Verify triple count from file header |
| 3.2.2 | Three entity types: papers (`scientificWork/`), datasets (`dataset/`), models (`model/`) | `build_paper_records.py` | — | `CLAUDE.md`, `pre_retrieval_methodology.md` | — | RDF [TODO] | None |
| 3.2.3 | ML-domain predicates: `mlso:hasTaskType`, `dcat:keyword`, etc. | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py` | — | `pre_retrieval_methodology.md` | — | RDF vocabularies [TODO] | Extract full predicate list from code |
| 3.3.1 | 280 questions; 265 answerable; fields: id, question, question_type, target_entity_iri, difficulty, is_answerable | — | — | `CLAUDE.md` | `data/questions/ml_questions_dataset.json` | — | Document is_answerable=False criterion; verify question_type taxonomy |
| 3.3.2 | Single gold IRI per question; closed-world assumption | — | — | `CLAUDE.md` | — | — | Document IRI normalisation rule |
| 3.3.3 | k=10 default; segmented by entity type, difficulty, question type | `src/retrieval/config.py` | — | `retrieval_stage_plan.md` | — | — | None |
| 3.4.1 | Raw RDF triples not suitable for dense retrieval due to IRI incoherence and linked-node indirection | — | — | `pre_retrieval_methodology.md` | — | Dense retrieval [TODO] | None |
| 3.4.2 | Two-pass streaming extraction; Pass 1 collects entity triples; Pass 2 resolves labels | `build_paper_records.py`, `build_dataset_records.py`, `build_model_records.py`, `shared/utils.py` | — | `pre_retrieval_methodology.md` | — | — | None |
| 3.4.3 | Canonical record decouples extraction from chunk building; field priority order for multi-predicate fields | All three `build_*_records.py` | — | v2 plan §4.3 | — | — | Verify priority order from code |
| 3.4.4 | Gold-first 200k paper subset; datasets and models used in full | `src/pre_retrieval/papers/raw/build_curated_subset.py` | `config/pre_retrieval_config.json` (max_papers=200000) | — | — | — | Verify full corpus size from `papers_master.jsonl` |
| 3.4.5.1 | 6 paper strategies; `enriched_metadata` is max-coverage (2,200 chars, 5+5+5+5+6+3 limits) | `src/pre_retrieval/papers/chunking/build_*.py` (6 files) | `config/pre_retrieval_config.json` | — | — | — | Verify `predicate_filtered` abstract limit is 500 chars in code (matches config) |
| 3.4.5.2 | 4 dataset strategies; title_only is minimal (512 chars); enriched_metadata is maximal (2,400 chars) | `src/pre_retrieval/datasets/chunking/` | `config/pre_retrieval_config.json` | — | — | — | None |
| 3.4.5.3 | 4 model strategies; `model_predicate_filtered` uses curated whitelist | `src/pre_retrieval/models/chunking/` | `config/pre_retrieval_config.json` | — | — | — | None |
| 3.4.6 | `all-MiniLM-L6-v2`; 384-dim; L2-normalised; ChromaDB HNSW; cosine metric; batch_size=64 | `src/pre_retrieval/shared/embedder.py`, `embed_and_store.py`, `vector_store.py` | `config/pre_retrieval_config.json` | `CLAUDE.md` | — | SBERT [TODO], ChromaDB [TODO], HNSW [TODO] | Verify total collection count; verify store size |
| 3.4.7 | Best representation per entity type selected by NDCG: enriched_metadata, dataset_title_only, model_predicate_filtered | — | `src/retrieval/config.py` (BEST_REPRESENTATION) | `CLAUDE.md` | `data/results/thesis_tables/best_per_entity.csv` | — | None (results in Chapter 4) |
| 3.5.1 | 6 retrieval methods in 3 families: Pure Semantic, Hybrid, Optional Fusion | `src/retrieval/pipeline.py`, `config.py` | — | `retrieval_stage_plan.md` | — | — | None |
| 3.5.2 | `pure_semantic_dense` passes pre-retrieval top-10 unchanged | `src/retrieval/dense_baseline.py` | `src/retrieval/config.py` | — | — | Dense retrieval [TODO] | None |
| 3.5.3.1 | `hybrid_type_filtering` filters by entity type IRI prefix; expected no-op (control) | `src/retrieval/filtering.py:run_type_filtering` | — | `retrieval_stage_plan.md` | — | — | None |
| 3.5.3.2 | `hybrid_type_onehop_filtering` boosts by one-hop richness: {tasks, datasets, methods, metrics, implementations} + "Linked Entities" bonus | `src/retrieval/filtering.py:run_hybrid_type_onehop_filtering`, `_onehop_richness` | — | — | — | — | None |
| 3.5.3.3 | `hybrid_predicate_aware_filtering` maps question_type to metadata field; boosted candidates sorted first | `src/retrieval/filtering.py:run_predicate_aware_filtering`, `_boost_by_predicate` | — | — | — | — | Confirm full question_type taxonomy coverage |
| 3.5.4.1 | `optional_rrf_fusion` uses formula `score = Σ 1/(60+rank)`, k=60; fusion groups: papers(3), datasets(2), models(2) | `src/retrieval/rrf.py:run_rrf_fusion`, `_rrf_score` | `src/retrieval/config.py` (RRF_K=60, RRF_FUSION_GROUPS) | — | — | RRF [TODO Cormack et al.] | None |
| 3.5.4.2 | `optional_rrf_symbolic` = RRF → type filter → predicate-aware boost (3-step composition) | `src/retrieval/rrf.py:run_rrf_symbolic` | — | — | — | — | None |
| 3.5.5 | `DEFAULT_TOP_K=10`; uniform output schema; k=10 defines recall ceiling | `src/retrieval/config.py`, `src/retrieval/pipeline.py`, `src/retrieval/result_schema.py` | — | — | — | — | Extract candidate schema fields from result_schema.py |
| 3.5.6 | Retrieval output feeds downstream post-retrieval stage; Hit@10 is maximum recoverability bound | `src/retrieval/result_schema.py` | — | — | `data/results/retrieval/` (output format) | RAG [TODO] | None |

---

## 7. Chapter 3 Citation Needs

The following citations are needed for the methodology chapter. All are marked as TODO pending verification of the exact references to use.

| Citation needed for | Placeholder | Notes |
|---|---|---|
| Retrieval-Augmented Generation | [TODO: Lewis et al. 2020 — verify title and venue] | Foundational RAG paper; cited in §3.1 and §3.5.6 |
| Resource Description Framework (RDF) | [TODO: W3C RDF specification] | Cite in §3.2.1 when describing N-Triples format |
| RDF triples and knowledge graph representation | [TODO: survey or foundational text] | Cite in §3.2.1–3.2.3 |
| Knowledge graph verbalization / KG-to-text | [TODO: survey on KG verbalisation] | Cited in §3.4.1 as related motivation |
| Dense retrieval / bi-encoder retrieval | [TODO: Karpukhin et al. DPR 2020 or equivalent] | Cited in §3.4.1, §3.5.2 |
| Sentence-BERT / SentenceTransformers | [TODO: Reimers & Gurevych 2019] | Cited in §3.4.6 when introducing the embedding model |
| all-MiniLM-L6-v2 model / MTEB benchmark | [TODO: HuggingFace model card; MTEB citation] | Cited in §3.4.6; include MTEB score if available |
| Vector databases / ChromaDB | [TODO: ChromaDB documentation or paper] | Cited in §3.4.6 |
| Approximate nearest-neighbour search | [TODO: survey or HNSW paper] | Cited in §3.4.6 when describing HNSW index |
| HNSW (Hierarchical Navigable Small World) | [TODO: Malkov & Yashunin 2018] | Cited in §3.4.6 |
| Hybrid retrieval (dense + sparse or symbolic) | [TODO: relevant survey or paper] | Cited in §3.5.3 when introducing the hybrid family |
| Symbolic retrieval / structured retrieval | [TODO: relevant survey or paper] | Cited in §3.5.3 |
| Reciprocal Rank Fusion | [TODO: Cormack, Clarke & Buettcher 2009 — verify] | Cited in §3.5.4.1 with the RRF formula |
| MLSea knowledge graph | [TODO: MLSea paper or dataset reference] | Cited in §3.2.1 when introducing the data source |

---

## 8. Chapter 3 Writing Checklist

Use this checklist before finalising the Chapter 3 draft.

### Pipeline Scope
- [ ] §3.1 introduces the full three-stage KG-RAG pipeline (pre-retrieval → retrieval → post-retrieval)
- [ ] Post-retrieval is presented as the third pipeline stage, not as a separate numbered section
- [ ] §3.5.6 connects retrieval output to post-retrieval using professional language (not defensive scope disclaimers)
- [ ] No sentence in Chapter 3 contains "not my responsibility", "out of scope for me", "handled by colleague", or equivalent

### Research Question Alignment
- [ ] Chapter 3 is explicitly aligned to RQ1 and RQ2
- [ ] RQ3 appears only as a downstream boundary reference (in §3.1 and §3.5.6) — no detailed RQ3 methodology
- [ ] Main research question is stated or referenced in §3.1

### Mandatory Layout
- [ ] Exact mandatory layout is used (no renamed sections, no added numbered post-retrieval section)
- [ ] All subsections 3.2.1–3.5.6 are present and in order
- [ ] §3.4.5 contains exactly three sub-subsections (3.4.5.1 papers, 3.4.5.2 datasets, 3.4.5.3 models)
- [ ] §3.5.3 contains exactly three sub-subsections (3.5.3.1, 3.5.3.2, 3.5.3.3)
- [ ] §3.5.4 contains exactly two sub-subsections (3.5.4.1, 3.5.4.2)

### MLSea and Dataset Description
- [ ] MLSea described: 26,606,202 triples, 6.4 GB, N-Triples format, three entity IRI namespaces
- [ ] Three entity types described with annotation density characterisation
- [ ] ML-domain predicates listed with Table 3.2
- [ ] Question dataset described: 280 total, 265 answerable, 15 excluded
- [ ] Question fields documented: `id`, `question`, `question_type`, `target_entity_iri`, `difficulty`, `is_answerable`
- [ ] Gold target entity design explained: single IRI per question, IRI normalisation, closed-world assumption

### Pre-Retrieval Methodology
- [ ] Two-pass streaming extraction explained with Figure 3.4
- [ ] Linked-entity label resolution explained (rdfs:label, foaf:name fallback to IRI tail)
- [ ] Canonical entity record fields documented with Table 3.6
- [ ] Gold-first 200k corpus curation explained with Table 3.7
- [ ] All 6 paper representation strategies described with character limits from config
- [ ] All 4 dataset representation strategies described with annotation sparsity observation
- [ ] All 4 model representation strategies described with predicate whitelist rationale
- [ ] `all-MiniLM-L6-v2`, 384-dim, L2-normalised, cosine, ChromaDB HNSW documented
- [ ] Pre-retrieval output section references best_per_entity.csv without presenting NDCG table (Chapter 4)

### Retrieval Methodology
- [ ] Design rationale for retrieval stage transition explained
- [ ] All 6 retrieval methods described in correct families (Pure Semantic, Hybrid, Optional Fusion)
- [ ] `pure_semantic_dense` described as passthrough of pre-retrieval top-10
- [ ] `hybrid_type_filtering` described as control/no-op method
- [ ] `hybrid_type_onehop_filtering` described with _ONEHOP_FIELDS: {tasks, datasets, methods, metrics, implementations}
- [ ] `hybrid_predicate_aware_filtering` described with question_type-to-predicate mapping
- [ ] RRF formula included: `score(d) = Σ 1/(k + rank_i(d))`, k=60
- [ ] RRF fusion groups documented: papers(3), datasets(2), models(2) from config.py
- [ ] `optional_rrf_symbolic` described as 3-step composition: RRF → type filter → predicate-aware
- [ ] k=10 default stated with justification; k defines recall ceiling
- [ ] Retrieval output connected to post-retrieval downstream use

### Terminology Consistency
- [ ] "pre-retrieval" (hyphenated, lowercase) used consistently
- [ ] "retrieval" (lowercase) used consistently
- [ ] "post-retrieval" (hyphenated, lowercase) used consistently
- [ ] "entity-centric representation" used consistently
- [ ] "canonical entity record" used consistently
- [ ] "chunk representation" used consistently
- [ ] "dense retrieval" used consistently
- [ ] "symbolic signal" used consistently
- [ ] "hybrid symbolic-semantic retrieval" used consistently
- [ ] "Reciprocal Rank Fusion" (capitalised, spelled out) used consistently with RRF acronym introduced once
- [ ] "top-10 candidates" (hyphenated) or "top 10 candidates" used consistently (pick one)

### Methodology/Results Separation
- [ ] No NDCG, Hit@1, Hit@5, Hit@10, or MRR values appear in methodology subsections (§3.4.1–§3.5.5)
- [ ] §3.4.7 mentions that best representations are selected by evaluation without reporting the values
- [ ] §3.5.6 does not include a method comparison table
- [ ] All numeric metric values are reserved for Chapter 4

### No SPARQL Retrieval Claims
- [ ] No claim that the system uses SPARQL for retrieval (extraction uses streaming N-Triples parsing, not SPARQL)
- [ ] Alternative (triple store + SPARQL) mentioned as rejected alternative in §3.4.2 only

### Figures and Tables
- [ ] All 9 figures planned (Figure 3.1–3.9)
- [ ] All 11 tables planned (Table 3.1–3.11)
- [ ] Each figure has: number, title, caption, LaTeX label, insert location
- [ ] Each table has: number, title, caption, LaTeX label, insert location

### Algorithms
- [ ] 11 algorithm boxes planned for Chapter 3 (Algorithms 1–11)
- [ ] No metric computation algorithm appears in Chapter 3
- [ ] RRF formula is in both Algorithm 10 and §3.5.4.1 prose

### Citations
- [ ] RAG citation placeholder included
- [ ] SBERT / SentenceTransformers citation placeholder included
- [ ] all-MiniLM-L6-v2 / MTEB citation placeholder included
- [ ] ChromaDB citation placeholder included
- [ ] HNSW citation placeholder included
- [ ] RRF citation placeholder included (Cormack et al. — verify)
- [ ] MLSea citation placeholder included

### Traceability
- [ ] Every methodology claim in traceability matrix (Section 6) is linked to at least one source file
- [ ] All TODOs from Section 9 are visible and actionable

---

## 9. Remaining TODOs

The following items were unresolved during the repository exploration phase. Each must be resolved before finalising the Chapter 3 thesis text.

### Data/Corpus TODOs
- [ ] **TODO-C1:** Verify the total number of paper records in `data/intermediate/raw_papers/papers_master.jsonl` (full paper count before the 200k cap). Needed for Table 3.7 "Corpus Construction Summary" — specifically the "Full corpus size" column for papers.
- [ ] **TODO-C2:** Verify the current total number of ChromaDB collections (documented as 18 in the v2 methodology plan — one per representation strategy). Confirm by inspecting `data/intermediate/chroma/` or querying the ChromaDB instance.
- [ ] **TODO-C3:** Verify the current size of the ChromaDB vector store (documented as 8.2 GB in the v2 plan). The actual size may differ as data has changed.

### Question Dataset TODOs
- [ ] **TODO-Q1:** Document the **difficulty assignment criteria** for the `difficulty` field (`easy`, `medium`, `hard`). The values appear in the data but the assignment methodology is not documented in any existing doc file. This is needed for §3.3.1 and §3.3.3.
- [ ] **TODO-Q2:** Document the **`is_answerable=False` criterion** — what makes a question unanswerable? Needed for §3.3.1 when stating that 15 of the 280 questions are excluded from metric computation.
- [ ] **TODO-Q3:** Verify the **full question type taxonomy** by inspecting `data/questions/ml_questions_dataset.json` and counting distinct `question_type` values. The taxonomy used in `src/retrieval/filtering.py` (task types, implementation types, year types, repository types, family types, keyword types) is verified from the code side but not confirmed against the actual question set.

### Implementation Consistency TODOs
- [ ] **TODO-I1:** Verify that the `predicate_filtered` strategy applies `abstract_max_characters=500` consistently in `src/pre_retrieval/papers/chunking/build_predicate_filtered_chunks.py`. The config specifies 500 but this should be confirmed in the builder script.
- [ ] **TODO-I2:** Extract the **candidate output schema** field list from `src/retrieval/result_schema.py` and use it to populate Table 3.11. The traceability matrix references `result_schema.py` but the fields have not been extracted.
- [ ] **TODO-I3:** Extract the **full predicate-to-field mapping** from all three `build_*_records.py` scripts and use it to populate Table 3.2. The mapping was partially reproduced in the v2 plan §4.3 but should be verified against the current code.

### Citation TODOs
- [ ] **TODO-CIT1:** Obtain exact citation for **RAG** (Lewis et al. 2020 — confirm title, venue, and year).
- [ ] **TODO-CIT2:** Obtain exact citation for **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher 2009 — confirm full title and venue).
- [ ] **TODO-CIT3:** Obtain exact citation for **SentenceTransformers / SBERT** (Reimers & Gurevych 2019 — confirm).
- [ ] **TODO-CIT4:** Obtain exact citation and **MTEB benchmark score** for `all-MiniLM-L6-v2`.
- [ ] **TODO-CIT5:** Obtain exact citation for **HNSW** (Malkov & Yashunin 2018 — confirm).
- [ ] **TODO-CIT6:** Obtain the **MLSea knowledge graph paper or dataset reference**.
- [ ] **TODO-CIT7:** Identify appropriate citations for: dense retrieval (DPR or equivalent), KG verbalization, vector databases (ChromaDB), hybrid retrieval, symbolic retrieval.

### Figure TODOs
- [ ] **TODO-F1:** Create **Figure 3.1** (KG-RAG pipeline overview) — must be drawn; no existing file.
- [ ] **TODO-F2:** Create **Figure 3.2** (MLSea RDF entity structure example) — extract sample N-Triples from `data/raw/pwc_1.nt` for a gold-target paper entity.
- [ ] **TODO-F3:** Create **Figure 3.3** (RDF triples to entity-centric chunk) — conceptual before/after diagram.
- [ ] **TODO-F4:** Create **Figure 3.4** (two-pass RDF extraction flow diagram).
- [ ] **TODO-F5:** Create **Figure 3.5** (canonical entity record construction schematic).
- [ ] **TODO-F6:** Create **Figure 3.6** (example chunk representations for `title_only`, `enriched_metadata`, `one_hop`) — extract from `data/intermediate/representations/papers/`.
- [ ] **TODO-F7:** Create **Figure 3.7** (embedding and vector indexing workflow).
- [ ] **TODO-F8:** Create **Figure 3.8** (retrieval method workflow — six methods from shared input).
- [ ] **TODO-F9:** Create **Figure 3.9** (top-k retrieval output and downstream handoff).

### Table TODOs
- [ ] **TODO-T1:** Populate **Table 3.7** (corpus construction summary) — full paper corpus size from `papers_master.jsonl` (TODO-C1), dataset corpus size, model corpus size.
- [ ] **TODO-T2:** Populate **Table 3.2** (RDF predicate-to-field mapping) — verify all predicate lists from `build_*_records.py`.
- [ ] **TODO-T3:** Populate **Table 3.11** (candidate output schema) — extract field list from `src/retrieval/result_schema.py`.
- [ ] **TODO-T4:** Populate **Table 3.3** (question dataset schema) with example values from `data/questions/ml_questions_dataset.json`.

---

*End of Chapter 3 Final Methodology Plan*
