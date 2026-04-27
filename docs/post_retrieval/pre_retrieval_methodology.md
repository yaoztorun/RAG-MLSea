# Pre-Retrieval Methodology

## Goal
Evaluate how RDF entities should be converted into text before embedding.

## Input
- Raw RDF: `data/raw/pwc_1.nt`
- Questions: `data/questions/ml_questions_dataset.json`

## Extraction
RDF triples are processed in two passes:
1. Identify entity subjects and collect their direct predicates.
2. Resolve labels/types for linked entities.

## Output Records
Papers:
- `paper_id`, `title`, `abstract`, `authors`, `keywords`, `tasks`, `datasets`, `methods`, `metrics`, `implementations`, `linked_entities`, `raw_predicates`, `publication_year`

Datasets:
- `dataset_id`, `label`, `description`, `issued_year`, `keywords`, `tasks`, `related_papers`, `related_implementations`, `linked_entities`, `raw_predicates`

Models:
- `model_id`, `label`, `description`, `tasks`, `datasets`, `related_papers`, `related_implementations`, `runs`, `metrics`, `linked_entities`, `raw_predicates`

## Representation Strategies

### Papers
- `title_only`
- `abstract_only`
- `title_abstract`
- `predicate_filtered`
- `enriched_metadata`
- `one_hop`

### Datasets
- `dataset_title_only`
- `dataset_metadata`
- `dataset_predicate_filtered`
- `dataset_enriched_metadata`

### Models
- `model_title_only`
- `model_metadata`
- `model_predicate_filtered`
- `model_enriched_metadata`

## Embedding
Each representation becomes one `source_text` string. The text is embedded using:

`sentence-transformers/all-MiniLM-L6-v2`

Vector dimension: 384  
Similarity: cosine  
Vector store: ChromaDB

## Query Matching
The user question is embedded with the same model as the entity representation.

Retrieval compares:

`question_embedding` ↔ `entity_representation_embedding`

The system does not compare the natural-language question directly against raw RDF triples.