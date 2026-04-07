# RAG-MLSea

Master thesis repository for offline MLSea / PwC retrieval and RAG experiments.

## Active architecture

The active repository is fully offline-first:

`data/raw/pwc_1.nt` → canonical paper records → representation files → embeddings / vector store → retrieval outputs → post-retrieval context building → generation / evaluation

Core rules:

- `data/raw/pwc_1.nt` is the source of truth
- no active stage depends on GraphDB or live SPARQL endpoints
- canonical paper records are built first and stored at `data/intermediate/raw_papers/papers_master.jsonl`
- retrieval writes offline result payloads to `data/retrieval_results/{representation}_results.json`
- post-retrieval consumes those retrieval outputs plus canonical records and representation rows

## Repository layout

```text
src/
  pre_retrieval/
    raw_papers/
    chunking/
    embeddings/
    retrieval/
    evaluation/
    scripts/
  retrieval/
    embedding/
    indexing/
    scripts/
    search/
  post_retrieval/
    generation/
    evaluation/
    pipeline/
    scripts/

docs/
  post_retrieval/
```

## Stage responsibilities

### 1. Pre-retrieval

Builds the offline artifacts derived from `pwc_1.nt`:

- canonical paper records
- representation files
- vector-store-ready documents

Primary entry points:

```bash
python -m src.pre_retrieval.scripts.run_build_records
python -m src.pre_retrieval.scripts.run_build_representations --representation title_only
python -m src.pre_retrieval.scripts.run_embed_store --representation title_only
python -m src.pre_retrieval.scripts.run_evaluate --representation title_only
```

### 2. Retrieval

Uses the existing offline experiment design to retrieve top-k papers for each question. Results are stored under `data/retrieval_results/`.

The current retrieval experiment design is unchanged.

### 3. Post-retrieval

Takes retrieval outputs that already contain:

- the question
- top-k retrieved papers
- `paper_id`
- optional `source_text`

Then it:

- loads canonical paper records from `data/intermediate/raw_papers/papers_master.jsonl`
- resolves representation text from `data/intermediate/representations/{representation}.jsonl` when needed
- filters / reranks the offline candidates
- builds a `<CONTEXT>` block for generation
- evaluates reranking or generated answers without any GraphDB/SPARQL dependency

Example entry points:

```bash
python -m src.post_retrieval.scripts.run_post_retrieval_pipeline --representation title_only --question-id mlsea_q_003
python -m src.post_retrieval.scripts.run_evaluate_retrieval --representation title_only
python -m src.post_retrieval.scripts.run_generate --representation title_only --question-id mlsea_q_003
```

## Representation strategies

1. `title_only`
2. `abstract_only`
3. `title_abstract`
4. `enriched_metadata`
5. `predicate_filtered`
6. `one_hop`

## Data

- `data/raw/pwc_1.nt` is local-only and not committed
- optional development input: `data/raw/pwc_1_sample.nt`
- generated artifacts live under `data/intermediate/`
- question datasets live under `data/questions/`

## Configuration

Pipeline defaults live in `config/pre_retrieval_config.json`.

Config covers:

- embedder backend
- embedding model name
- Chroma mode / host / port / persistence path
- evaluation top-k values
- representation-specific text limits

Default vector-store settings use Chroma HTTP mode:

```json
"vector_store": {
  "provider": "chroma",
  "chroma_mode": "http",
  "chroma_host": "localhost",
  "chroma_port": 8000,
  "persist_directory": "data/intermediate/chroma"
}
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Chroma before embedding or retrieval evaluation:

```bash
chroma run --path data/intermediate/chroma
```

## Key offline outputs

- `data/intermediate/raw_papers/papers_master.jsonl`
- `data/intermediate/raw_papers/extraction_stats.json`
- `data/intermediate/raw_papers/predicate_stats.json`
- `data/intermediate/representations/*.jsonl`
- `data/intermediate/representations/*_stats.json`
- `data/retrieval_results/*_results.json`
- `data/retrieval_results/summary.json`
- `data/retrieval_results/summary.md`
- `data/retrieval_results/summary.csv`

## Notes

- the active pipeline no longer uses GraphDB or SPARQL for post-retrieval
- legacy generated artifacts such as embedding dumps and evaluation logs should stay out of version control
- post-retrieval strategy notes now live in `docs/post_retrieval/`
