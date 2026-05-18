# Thesis Overview — RAG over MLSea Knowledge Graphs

## Thesis
Retrieval-Augmented Generation over Machine Learning Knowledge Graphs.

## Core Problem
Standard RAG retrieves from text documents, but MLSea is an RDF knowledge graph. Entity information is distributed across predicates and linked nodes, so the main challenge is how to convert graph entities into retrievable text representations.

## Pipeline
1. Pre-retrieval: build and evaluate entity representations.
2. Retrieval: compare candidate generation strategies.
3. Post-retrieval: filter, re-rank, construct context, generate answers.

## Main Claim
There is no universal best representation across all KG entity types. Papers, datasets, and models require different representation strategies.

## Entity Types
- Papers: text-rich; title, abstract, metadata.
- Datasets: sparse; label/title often strongest.
- Models: graph-heavy; repository links and linked entities matter.

## Current Best Representations
- Paper: `enriched_metadata`
- Dataset: `dataset_title_only`
- Model: `model_predicate_filtered`

## Key Insight
A combined `enriched + predicate` representation may help hard/multi-hop queries, but can introduce noise for simple queries. This is an important thesis observation.