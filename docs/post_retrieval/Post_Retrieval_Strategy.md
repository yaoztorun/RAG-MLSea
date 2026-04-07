# Offline Post-Retrieval Strategy

The post-retrieval stage now starts **after** the active retrieval pipeline has already produced top-k results from offline representations.

## Inputs

The offline contract is:

- canonical paper records from `data/intermediate/raw_papers/papers_master.jsonl`
- retrieval outputs from `data/retrieval_results/{representation}_results.json`
- optional representation rows from `data/intermediate/representations/{representation}.jsonl`

No step in this stage talks to GraphDB or sends SPARQL queries.

## Pipeline phases

1. **Load canonical records**
   - build a `paper_id -> canonical record` lookup from `papers_master.jsonl`
   - treat that lookup as the source of truth for paper metadata

2. **Resolve retrieved context**
   - take the retrieval stage's top-k results for a question
   - use the returned `paper_id` values to attach canonical metadata
   - reuse `source_text` from retrieval outputs when present, or resolve it from the matching representation file

3. **Hard filtering**
   - drop low-confidence retrieval hits using the retrieval score threshold
   - short-circuit if nothing survives

4. **Cross-encoder re-ranking**
   - re-rank the offline candidates with the MS MARCO cross-encoder
   - preserve the ablation option to skip the cross-encoder when needed

5. **Generation context formatting**
   - serialize only the highest-ranked candidates into a `<CONTEXT>` block
   - include canonical metadata such as title, year, authors, tasks, keywords, and abstract
   - include the retrieved representation text so generation stays aligned with the retrieval view of the document

## Output contract

The post-retrieval pipeline produces:

- a filtered and optionally re-ranked candidate list
- a `<CONTEXT>` string for answer generation
- no direct dependency on GraphDB, SPARQL endpoints, or live graph services
