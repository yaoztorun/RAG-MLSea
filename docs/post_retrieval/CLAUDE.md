# CLAUDE.md — RAG over MLSea Knowledge Graphs

## Project
Thesis: Retrieval-Augmented Generation over Machine Learning Knowledge Graphs  
Author: Yiğit Alp Öztorun  
Embedder: `sentence-transformers/all-MiniLM-L6-v2`  
Vector store: ChromaDB  
Status: pre-retrieval complete; retrieval stage being added; post-retrieval scaffold exists.

## Core Thesis Idea
This project builds an end-to-end KG-RAG pipeline over MLSea RDF data from Papers with Code.

Pipeline stages:
1. **Pre-retrieval**: convert RDF entities into text representations, embed, evaluate representation quality.
2. **Retrieval**: compare candidate generation strategies using existing pre-retrieval top-K outputs.
3. **Post-retrieval**: filter, re-rank, build context, generate and evaluate answers.

Key finding so far:
There is no universal best representation across papers, datasets, and models. Best representation is entity-dependent.

## Entity Types
- Paper: text-rich; title + abstract + metadata.
- Dataset: sparse; label/title often strongest.
- Model: graph-heavy; repository and linked-entity context matters.

Best pre-retrieval representations:
- Paper: `enriched_metadata`
- Dataset: `dataset_title_only`
- Model: `model_predicate_filtered`

Important thesis insight:
A combined `enriched + predicate` representation may help hard/multi-hop questions but can introduce noise for simple queries. Treat this as an analysis finding, not automatically as final method.

## Current Data Flow
Raw RDF:
- `data/raw/pwc_1.nt`

Questions:
- `data/questions/ml_questions_dataset.json`
- 280 questions
- Fields include: `id`, `question`, `question_type`, `target_entity_iri`, `answer`, `text_answer`, `difficulty`, `is_answerable`

Intermediate records:
- `data/intermediate/raw_papers/papers_master.jsonl`
- `data/intermediate/raw_papers/papers_subset_200k.jsonl`
- `data/intermediate/raw_datasets/`
- `data/intermediate/raw_models/`

Representations:
- `data/intermediate/representations/papers/`
- `data/intermediate/representations/datasets/`
- `data/intermediate/representations/models/`

Pre-retrieval results are stored under:
- `data/results/pre_retrieval_results/`
  (the code also checks `data/results/pre_retrieval/` and `data/retrieval_results/` as fallbacks)

Retrieval-stage results should be stored under:
- `data/results/retrieval/`

Post-retrieval results should be stored under:
- `data/results/post_retrieval/`

If old paths exist under `data/retrieval_results/`, treat them as legacy pre-retrieval outputs and migrate/update paths carefully.

## Representation Strategies

### Papers
- `title_only`
- `abstract_only`
- `title_abstract`
- `predicate_filtered`
- `enriched_metadata`
- `one_hop`

Best: `enriched_metadata`

### Datasets
- `dataset_title_only`
- `dataset_metadata`
- `dataset_predicate_filtered`
- `dataset_enriched_metadata`

Best: `dataset_title_only`

### Models
- `model_title_only`
- `model_metadata`
- `model_predicate_filtered`
- `model_enriched_metadata`

Best: `model_predicate_filtered`

## Pre-Retrieval Evaluation
For each question:
1. Embed question with same SentenceTransformer model.
2. Compare question embedding with entity representation embeddings in ChromaDB.
3. Retrieve top-10.
4. Check whether `target_entity_iri` appears.

Metrics:
- Hit@1
- Hit@5
- Hit@10
- MRR
- NDCG

Primary metric: NDCG.

Use segmentation by:
- entity type
- difficulty
- question type/category

## Current Pre-Retrieval Results Summary

Best per entity:
- Paper `enriched_metadata`: NDCG 0.8225, Hit@1 0.7753
- Dataset `dataset_title_only`: NDCG 0.3822, Hit@1 0.2807
- Model `model_predicate_filtered`: NDCG 0.8750, Hit@1 0.8000

Important observations:
- Papers benefit from enriched metadata.
- Datasets remain difficult because metadata is sparse.
- Models benefit strongly from predicate filtering because sparse/noisy model records are excluded.
- Abstract-only performs poorly because title signal is removed.
- Hard paper questions benefit most from `enriched_metadata`.
- Easy/medium paper questions can perform better with simpler or predicate-based representations.

## Retrieval Stage Plan

Create/maintain:
```text
src/retrieval/
  __init__.py
  config.py
  data_loading.py
  result_schema.py
  dense_baseline.py
  filtering.py
  rrf.py
  evaluate_retrieval_stage.py
  aggregate_retrieval_stage.py
  pipeline.py
  README.md
  scripts/
    run_retrieval_stage.py
    run_evaluate_retrieval_stage.py
    run_all_retrieval_experiments.py