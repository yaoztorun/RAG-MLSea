# ML Questions Dataset Audit Report

Original dataset: 280 questions

## Audit Results
- KEEP (valid, no changes): 0
- REPAIR (valid, schema fixed): 110
- REPLACE (property→entity-retrieval): 51
- DISCARD (removed): 119

## Discard Reasons
- property question; no replacement possible: 64
- replacement has duplicate text: 27
- property question; IRI already used: 13
- IRI already used twice: 6
- off-topic unanswerable: 5
- IRI not in canonical (http://w3id.org/mlsea/pwc/dataset/2018 n2c2): 2
- IRI not in canonical (http://w3id.org/mlsea/pwc/dataset/10,000 People): 1
- IRI not in canonical (http://w3id.org/mlsea/pwc/dataset/OCR): 1

## Example Replaced Questions
- mlsea_q_001: property question replaced with dataset_title_to_entity
- mlsea_q_014: property question replaced with dataset_task_to_dataset
- mlsea_q_017: property question replaced with dataset_task_to_dataset
- mlsea_q_018: property question replaced with dataset_task_to_dataset
- mlsea_q_026: property question replaced with dataset_title_to_entity

## Example Discarded Questions
- mlsea_q_002: property question; no replacement possible
- mlsea_q_003: property question; no replacement possible
- mlsea_q_004: property question; no replacement possible
- mlsea_q_005: property question; no replacement possible
- mlsea_q_006: property question; no replacement possible

## Final Distribution
- Total: 520
- paper: 257
- model: 132
- dataset: 131
