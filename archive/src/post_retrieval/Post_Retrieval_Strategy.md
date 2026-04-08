# Post-Retrieval Optimization Strategy for RAG 

This document outlines the optimal "Retrieve, Score, Re-Rank, and Format" strategy for processing MLSea-KG data using SBERT before passing it to a Large Language Model (e.g. LLaMA) for generation.

## 1. The Challenge
If you pass too many retrieved raw documents or RDF triples directly to an LLM, two major issues occur:
1. **Lost in the Middle Syndrome:** LLMs tend to forget or overlook information stored in the middle of long prompts.
2. **Dense Retrieval Inaccuracies:** SBERT (a Bi-Encoder) is extremely fast but only measures cosine similarity. This means it often retrieves documents that use similar vocabulary but have completely different semantic meanings. 

To boost accuracy, we insert an aggressive four-phase **Post-Retrieval** pipeline before generation.

## Phase 1: Bi-Encoder Retrieval (The Fast Pass)
You embed the natural language question using a lightweight Bi-Encoder (`all-MiniLM-L6-v2`). You calculate the cosine similarity against your pre-computed FAISS index or GraphDB embeddings. 
**Action:** Retrieve the **Top 20** or **Top 50** broadly related candidates. 

## Phase 2: Score-Based Hard Filtering
Not all queries have successful hits. If the user asks an unanswerable question or something completely outside the scope of the KG, SBERT will still try to find the "closest" match, which will likely be garbage.
**Action:** Apply a strict minimum cosine limit (e.g., `cosine_similarity > 0.20`). If no candidates pass this filter, *short-circuit* the pipeline. Do not ping the LLM. Simply return: "The question is unanswerable from the available MLSea knowledge graph context."

## Phase 3: Semantic Re-Ranking (The Cross-Encoder)
This is the most critical intervention for cutting down hallucinations.
A Cross-Encoder (such as `cross-encoder/ms-marco-MiniLM-L-6-v2`) does not use cosine similarity. Instead, it feeds the user's question and the retrieved text *simultaneously* into the transformer model. This allows the model's self-attention layers to perfectly align the specific words in the question to the words in the context, outputting highly accurate "relevance scores" (from 0 to 1).
**Action:** Take the 20 surviving chunks from Phase 2, score all of them using the Cross-Encoder, and re-sort them based on the new Cross-Encoder score. 

## Phase 4: Top-K Context Selection and Formatting Let down
Now that your candidates are perfectly re-ranked, you must solve the context-window limitation. 
**Action:** Truncate the results strictly to the absolute **Top 3 or Top 5** hits. 

LLMs struggle with raw, disjointed RDF triples. You must serialize them into an easy-to-read schema. Structure the remaining items clearly with metadata inside a labeled prompt block. 

### Final LLM Prompt Example:
```text
Answer the user's question relying strictly on the following context.

<CONTEXT>
--- Result 1 ---
Entity IRI: http://w3id.org/mlsea/pwc/scientificWork/Sensor-Independent
Title: Sensor-Independent Illumination Estimation for DNN Models 
Keywords: Illumination Estimation; Computer Vision

--- Result 2 ---
...
</CONTEXT>

QUESTION: What are the keywords associated with Sensor-Independent Illumination Estimation?
```

## Phase 5: Generation via Local LLMs (LLaMA / VSC Integration)
In this project, the intended vision relies on running open-source LLMs natively, specifically targeting the **LLaMA** family of models, rather than exclusively depending on closed APIs like OpenAI. 
Because the experiments will be scaled using the multi-GPU memory architecture of the **Vlaams Supercomputer Centrum (VSC)**, the generation phase skips API wrappers (like Ollama) and relies directly on the HuggingFace `transformers` and `accelerate` libraries.

**Action:** The `<CONTEXT>` block from Phase 4 is injected perfectly into the `System Message` of a LLaMA chat template (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`). The model is loaded onto VSC GPU memory, evaluates the structured chunking, and produces the final deterministic answer.

## Why this Architecture Wins
By utilizing **SBERT for speed** (to cut millions of entities down to 20) and a **Cross-Encoder for accuracy** (to perfectly surface the Top 3), you guarantee that the LLaMA model receives the absolute highest-quality subset of information, minimizing VSC token processing waste and strictly anchoring truthfulness.

## How to Execute the Pipeline (Step-by-Step Guide)
So you don't forget when spinning this back up for development or for the VSC deployment, follow these exact steps to run the post-retrieval and generation scripts:

### Requirement Checklist
1. **GraphDB is Running:** Ensure your local GraphDB repository is active on `http://localhost:7200`. If it's closed, the SPARQL data pull will fail instantly.
2. **Environment Variables:** Ensure you are using the correct Python environment containing `sentence-transformers`, `torch`, `SPARQLWrapper`, and `transformers`.
3. **HuggingFace Authenticated (For Phase 5):** You must be logged in via `huggingface-cli login` in your terminal to download gated Meta LLaMA 3 models.

### Execution 1: Testing the Post-Retrieval Pipeline locally
If you only want to test the extraction, chunking, and Cross-Encoder Re-Ranking (without the heavy LLM generation):
```bash
python post_retrieval_pipeline.py
```
**Expected Output:** This will print the `--- Result 1 ---` to `--- Result 3 ---` chunks perfectly formatted inside `<CONTEXT>` tags.

### Execution 2: Testing the LLM Generation (VSC / High-Memory Environment)
If you want to test the full End-to-End RAG system (passing the Re-Ranked context to LLaMA 3):
```bash
python llama_generation.py
```
**Expected Output:** This script automatically imports the logic from `post_retrieval_pipeline.py`, fetches the graph context, passes it securely into LLaMA 3's System Prompt via HuggingFace's chat templates, and prints the final deterministic Machine Learning answer.
*(Note: If you run this script locally without sufficient GPU VRAM, it will easily crash. It is fully designed to scale onto SLURM-backed VSC consumer GPUs).*
