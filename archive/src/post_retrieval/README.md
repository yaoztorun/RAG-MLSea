# Post-Retrieval Phase Architecture

This module contains the codebase for the **Post-Retrieval, Filtering, Re-Ranking, and Generation Integration** phases of the MLSea-KG RAG (Retrieval-Augmented Generation) pipeline.

## System Workflow
The pipeline acts as the crucial intermediary layer connecting the raw `GraphDB` vector search output to the final **LLaMA 3** natural language generator. It perfectly executes the following architectural phases:

### 1. The Dataset (`data/questions/ml_questions_dataset.json`)
The foundation of the evaluation is a highly-curated JSON dataset containing exactly 50 machine learning queries (45 semantic queries, 5 strictly unanswerable). They map concepts like `Authors`, `Publication Year`, `Keywords`, and `Tasks` to their structural MLSea IRIs.

### 2. SBERT Bi-Encoder Retrieval (The Dragnet)
The first execution step uses `all-MiniLM-L6-v2` to query the Knowledge Graph. It converts incoming text to 384-dimensional dense vectors to quickly retrieve the Top 20 broadly-related `mlso:ScientificWork` nodes based purely on cosine similarity.

### 3. Score-Based Hard Filtering
To prevent complete hallucinations from LLaMA running on unanswerable inputs (e.g. Question 21: *'How is the weather today?'*), the pipeline enforces a mathematically strict Trapdoor Protocol:
- If `cosine_similarity < 0.20`, the pipeline forcibly short-circuits.
- Returns deterministic response: *"The question is unanswerable from the available MLSea knowledge graph context."*

### 4. Semantic Re-Ranking (MS-MARCO Cross-Encoder)
Because SBERT determines relationship scores without allowing specific query terms to self-attend to the specific document terms, it lacks high-precision judgment. We pass the surviving Top-20 Candidates into an MS-MARCO `Cross-Encoder` model:
- The question and document interact symmetrically.
- Candidates are natively re-scored and flawlessly sorted based on highest relevance confidence.
- Results perfectly overcome 'Lost in the middle' syndrome by truncating down to an absolute **Top-3 Context Selection**.

### 5. LLaMA/VSC Generation Target (`src/generation/llama_generation.py`)
The re-ranked Top 3 contexts are strictly serialized into a `<CONTEXT>` HTML-style encapsulation.
This output string is safely ferried directly into the `System Message` of a LLaMA chat-instruction template format. This ensures that the LLaMA model, deployed robustly over the **Vlaams Supercomputer Centrum (VSC)** cluster, functions purely as an unhallucinating synthesizer rather than an untethered predictor.
