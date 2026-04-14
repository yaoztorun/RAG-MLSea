# Offline Post-Retrieval Strategy

The post-retrieval stage covers the journey from raw retrieval results to a final, evaluated answer.

## Inputs

The offline contract is:
- canonical paper records from `data/intermediate/raw_papers/papers_master.jsonl`
- retrieval outputs from `data/retrieval_results/{representation}_results.json`
- optional representation rows from `data/intermediate/representations/{representation}.jsonl`

## Pipeline Phases

1. **Load canonical records**
   - Build a `paper_id -> canonical record` lookup from `papers_master.jsonl`.

2. **Resolve retrieved context**
   - Resolve metadata and representation text for the top-k results.

3. **Hard filtering**
   - Drop low-confidence retrieval hits using the retrieval score threshold.

4. **Cross-encoder re-ranking**
   - Re-rank candidates using a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).

5. **Generation context formatting**
   - Serialize the highest-ranked candidates into a `<CONTEXT>` block.

6. **Answer Evaluation**
   - **Quantitative**: Calculate SAS (Semantic Answer Similarity) and ROUGE-L.
   - **Qualitative (LLM-as-a-Judge)**: Use a separate LLM pass to verify factual correctness against the ground truth.
   - **Audit Trail**: Save raw judge responses for manual spot-checking and verification.

## Output Contract

The pipeline produces:
- A final generated answer.
- A suite of metrics (SAS, ROUGE, Judge Accuracy).
- A trace of the judge's reasoning for every question.
- No direct dependency on GraphDB or live graph services.
