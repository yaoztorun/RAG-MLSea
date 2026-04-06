# RAG-MLSea

Master thesis repository for pre-retrieval and RAG experiments on MLSea / PwC paper metadata.

## Active pre-retrieval pipeline

The active pipeline is now offline-first:

`data/raw/pwc_1.nt` → canonical paper records → representation files → one-time embeddings → persistent Chroma collections → retrieval evaluation

Key rules:

- `data/raw/pwc_1.nt` is the source of truth for paper metadata
- the active pipeline does not depend on live GraphDB / SPARQL queries
- one embedding model is used across all representation strategies
- document embeddings are stored once and reused
- only questions are embedded at runtime

## Active pipeline layout

```text
src/pre_retrieval/
  raw_papers/
    inspect_paper_predicates.py
    build_paper_records.py
  chunking/
    build_representations.py
  embeddings/
    embed_and_store.py
    vector_store.py
  retrieval/
    retrieve.py
  evaluation/
    evaluate_retrieval.py
  scripts/
    run_build_records.py
    run_build_representations.py
    run_embed_store.py
    run_evaluate.py
```

Archived GraphDB / SPARQL assets live under `src/pre_retrieval/legacy/graphdb/`.

## Representation strategies

Run and evaluate these progressively:

1. `title_only`
2. `abstract_only`
3. `title_abstract`
4. `enriched_metadata`
5. `one_hop`

The first complete baseline is `title_only`.

## Configuration

Pipeline defaults live in `/home/runner/work/RAG-MLSea/RAG-MLSea/config/pre_retrieval_config.json`.

Config covers:

- embedding model name
- Chroma database path
- evaluation top-k values
- max text lengths per representation

## Setup

Install the offline pipeline dependencies in your environment:

```bash
pip install rdflib sentence-transformers chromadb numpy
```

Place the local RDF dump at `data/raw/pwc_1.nt`, or pass `--nt-path` explicitly.

## Baseline workflow

Build canonical paper records:

```bash
python -m src.pre_retrieval.scripts.run_build_records
```

Build the first representation baseline:

```bash
python -m src.pre_retrieval.scripts.run_build_representations --representation title_only
```

Embed once and persist in Chroma:

```bash
python -m src.pre_retrieval.scripts.run_embed_store --representation title_only
```

Evaluate retrieval:

```bash
python -m src.pre_retrieval.scripts.run_evaluate --representation title_only
```

To generate all scaffolded representations:

```bash
python -m src.pre_retrieval.scripts.run_build_representations --representation all
```

## Data outputs

The pipeline writes to:

- `data/intermediate/raw_papers/papers_master.jsonl`
- `data/intermediate/raw_papers/predicate_stats.json`
- `data/intermediate/representations/*.jsonl`
- `data/intermediate/chroma/`
- `data/intermediate/retrieval_results/*_results.json`

## Migration notes

- active execution should use the new `src/pre_retrieval/scripts/` entry points
- legacy GraphDB / SPARQL extraction code has been removed from the active path but kept for reference
- the offline extractor builds canonical paper records first, then derives representation variants from those records
