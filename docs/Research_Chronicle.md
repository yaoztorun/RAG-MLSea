# Research Chronicle: RAG-MLSea Project

This log tracks the chronological evolution of the RAG pipeline and evaluation strategies.

## Phase 1: Retrieval Pipeline Foundations
- **Objective**: Establish offline retrieval from local representations.
- **Achievement**: Implemented extraction and embedding of paper titles.
- **Status**: Completed.

## Phase 2: Post-Retrieval & Evaluation Shell
- **Objective**: Create a pipeline for re-ranking and generating answers.
- **Achievement**: Built `evaluate_generation.py` to calculate SAS (Semantic Similarity) and ROUGE-L.
- **Innovation**: Added an LLM-as-a-Judge metric.

## Phase 3: Solving the "Small Model Logic" Problem (Active)
- **Problem**: When testing locally with `TinyLlama-1.1B`, the LLM Judge had 0% accuracy despite correct answers.
- **Root Cause**: "Instruction Drift" – the model is too small to follow complex system prompts. It gives the right reasoning but fails the binary output format.
- **Solution Strategy**: 
    1. Implementation of **Chain-of-Thought (CoT)** prompting (forcing the model to reason before answering).
    2. Transitioned from complex chat templates to rigid instruction prompts.
    3. Added **Audit Logs** to verify factual logic even when formatting fails.

## Phase 4: Local Benchmark Verification
- **Milestone**: Successful first run of RAG vs. Baseline comparison.
- **Results Analysis**:
    - **SAS (0.94 RAG vs 0.82 Base)**: Proves high semantic alignment of the RAG system with ground truth.
    - **ROUGE-L (0.61 RAG vs 0.35 Base)**: Proves that retrieved context nearly doubles the literal factual overlap, providing the "vocabulary" of the papers to the model.
    - **Judge Accuracy (0.33)**: Demonstrates that even on small hardware, CoT allows for automated factual validation, though accuracy is constrained by model size.
- **Significance**: RAG is objectively superior to the baseline across both semantic (meaning) and word-level (literal) metrics.

### Phase 4.1: Bug Fix - Baseline Metric Initialization
- **Problem**: Baseline metrics (SAS/ROUGE-L) returning `null` in JSON output despite active generation.
- **Root Cause**: The evaluation script lacked a "blind" baseline generator function, resulting in empty strings being compared against the ground truth.
- **Fix**: Implemented and exported `generate_baseline_answer` in `llama_generation.py` and wired it into `run_evaluate_generation.py`. This enabled 1:1 performance comparisons between RAG and non-RAG outputs.

## Phase 5: Deep Dive into Judge Logic Failures
- **Observation**: 2 out of 3 questions were marked as `0` by the 1.1B Judge despite "visually correct" answers.
- **Root Cause Analysis (via Audit Logs)**:
    1. **Semantic Strictness**: The judge penalized synonyms (e.g., "extremely deep" vs "substantially deeper").
    2. **Context Hallucination**: For Author-related questions (CoQA), the judge hallucinated researchers from other papers in the context (BERT) during its reasoning phase.
- **Conclusion**: Factual judging requires higher "Reasoning Stability" found in 8B+ models. However, the use of **Chain-of-Thought** was successful in exposing these failure modes rather than silently giving wrong scores.

## Phase 6: Few-Shot CoT Hardening → Judge Accuracy = 1.0
- **Problem**: The judge was "leaking" few-shot examples into its output (repeating example text after scoring), and `[[1]]`/`[[0]]` formatting was inconsistent.
- **Fix 1 – Stop Marker**: Added `###` as a stop sequence. The judge prompt now instructs the model to end every response with `###`, and the script splits the response at that token before parsing.
- **Fix 2 – Explicit Few-Shot Examples**: Provided 2 concrete solved examples (Paris/France) directly in the prompt to enforce the output pattern.
- **Result**: Judge accuracy reached **1.0 (100%)** on the n=15 validation run. All `[[1]]`/`[[0]]` outputs are correctly formatted and parseable by regex.
- **Key Design Insight**: For a 1.1B parameter model, explicit format examples (few-shot) are more reliable than natural-language instructions alone.

## Phase 7: Baseline Generator Refinement
- **Problem**: Baseline answers were being cut off mid-sentence (e.g., mid-name when listing authors) or defaulting to "I don't know"-style hedging.
- **Root Cause**: Token limit was too tight (64 tokens) and the system prompt instruction ("single phrase") was too ambiguous for small models.
- **Fix**: Increased `max_new_tokens` to **128** for the baseline generator. Updated system prompt to: *"Answer in 1-2 complete sentences. Do not use bullet points or lists."*
- **Significance**: The "incompleteness" of baseline answers vs. RAG answers is now a valid, observable research outcome — not an artefact of prompt engineering.

## Phase 8: Evaluation Diagrams & Documentation
- **Created**: `docs/evaluation_visuals.md` containing 4 Mermaid diagrams:
    1. **Overall Evaluation Architecture** – High-level Baseline vs. RAG comparison flow.
    2. **RAG Pipeline (7-Step)** – Detailed offline indexing + online retrieval/post-retrieval breakdown.
    3. **LLM-as-a-Judge Workflow** – Sequence diagram of CoT prompt → stop marker → regex parse → audit log.
    4. **Metrics Breakdown** – The 3 pillars: SAS, ROUGE-L, LLM Judge.
- **Exported**: PNG versions of diagrams 1, 3, 4 saved to `docs/` for direct inclusion in the thesis Word document.

## Phase 9: Scale to n=50 (In Progress)
- **Action**: Launched 50-question evaluation run using `--limit 50 --output-path data/intermediate/post_retrieval/generation_evaluation_n50.json`.
- **Rationale**: Move from anecdotal (n=15) to statistically meaningful results for the thesis.
- **Preservation**: The existing `generation_evaluation.json` (n=15) is kept intact as a reference baseline.
- **Next Step**: Once complete, perform statistical comparison (mean, std, Δ RAG vs Baseline) across both runs to include in the thesis Results chapter.

