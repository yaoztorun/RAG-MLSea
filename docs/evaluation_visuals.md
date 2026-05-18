# RAG Pipeline Evaluation Visualizations

These Mermaid diagrams visually explain the evaluation framework. You can screenshot these or recreate them directly in Google Slides for your presentation.

## 1. Overall Evaluation Architecture (Baseline vs. RAG)

This diagram shows how we scientifically prove the value of the RAG system by comparing its output against a "Baseline" (no context) generation, and scoring both against the Ground Truth.

```mermaid
graph LR
    Q[Input Question] --> BaseGen[Baseline Generator<br>No Context]
    Q --> RAG_Pipeline[["RAG Pipeline<br>(see Diagram 2)"]]

    GT[(Ground Truth<br>Answers Dataset)] --> Eval
    BaseGen --> |Baseline Answer| Eval
    RAG_Pipeline --> |RAG Answer| Eval

    subgraph Evaluation ["Evaluation Engine (LLMOps)"]
        Eval[Score Calculation]
        Eval --> SAS[SAS<br>Semantic Similarity]
        Eval --> ROUGE[ROUGE-L<br>Literal Overlap]
        Eval --> Judge[LLM-as-a-Judge<br>Factual Accuracy]
    end

    SAS --> JSON[(generation_evaluation.json)]
    ROUGE --> JSON
    Judge --> JSON

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef highlight fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef metric fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef db fill:#eceff1,stroke:#455a64,stroke-width:2px;
    classDef pipeline fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;

    class Eval,BaseGen highlight;
    class SAS,ROUGE,Judge metric;
    class GT,JSON db;
    class RAG_Pipeline pipeline;
```

## 2. RAG Pipeline – Detailed 7-Step Architecture

This diagram breaks down the internal steps of the RAG Pipeline, from offline indexing of the MLSea Knowledge Graph all the way to the serialized context handed off to the LLM generator.

```mermaid
graph LR
    Q[Input Question] --> QEnc

    subgraph RAG_Pipeline ["RAG Pipeline (7 Steps)"]
        direction TB

        subgraph Indexing ["① Offline Indexing"]
            direction LR
            MLSea[(MLSea<br>Knowledge Graph)] --> Chunk["② SPARQL Chunking<br>(Extract & Aggregate)"]
            Chunk --> Embed["③ Bi-Encoder<br>Embedding (sBERT)"]
            Embed --> VecDB[("④ Vector Index<br>(FAISS)")]
        end

        subgraph Online ["⑤⑥⑦ Online Retrieval & Post-Retrieval"]
            direction LR
            QEnc["⑤ Query Encoding<br>(sBERT)"] --> CosSim["⑥ Cosine Similarity<br>Search → Top-20"]
            CosSim --> Filter{"Score Filter<br>cosine > 0.20?"}
            Filter -- Yes --> ReRank["⑦ Cross-Encoder<br>Re-Ranking (MS-MARCO)"]
            Filter -- No --> Unans["Unanswerable"]
            ReRank --> Ctx["Top-3 Context<br>Serialized"]
        end

        VecDB -. similarity search .-> CosSim
    end

    Ctx --> RAGGen[RAG Generator<br>With Context]
    Unans --> RAGGen

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef highlight fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef indexing fill:#e3f2fd,stroke:#1565c0,stroke-width:1px;
    classDef db fill:#eceff1,stroke:#455a64,stroke-width:2px;
    classDef logic fill:#fff9c4,stroke:#f9a825,stroke-width:2px,stroke-dasharray:5 5;

    class RAGGen highlight;
    class Chunk,Embed,QEnc indexing;
    class MLSea,VecDB db;
    class Filter logic;
```

## 3. LLM-as-a-Judge Workflow & Audit Trail

This diagram illustrates the engineered flow for the LLM judge, specifically highlighting how we solved the "Instruction Drift" issue using Chain-of-Thought and robust parsing.

```mermaid
sequenceDiagram
    participant Pipeline as Eval Script
    participant Prompt as Prompt Builder
    participant LLM as TinyLlama-1.1B Judge
    participant Log as JSON Audit Trail

    Pipeline->>Prompt: Provide Ground Truth & Generated Answer
    Prompt->>Prompt: Inject Few-Shot Examples
    Prompt->>Prompt: Add Chain-of-Thought Instructions
    Prompt->>LLM: Send Full Prompt
    
    Note over LLM: Model processes context<br>and generates reasoning
    
    LLM-->>Pipeline: Raw Response (e.g., "Reasoning... [[1]] ###")
    
    Pipeline->>Pipeline: Split at '###' Stop Marker
    Pipeline->>Pipeline: Regex Parsing to extract [[1]] or [[0]]
    
    Pipeline->>Log: Save Binary Score (Accuracy)
    Pipeline->>Log: Save Raw Reasoning (Observability)
```

## 4. Metrics Breakdown

A simple flow showing exactly what each metric measures.

```mermaid
graph LR
    subgraph "Generated Answer vs Ground Truth"
        A[Generated Answer] --> Comp[Comparison]
        GT[Ground Truth] --> Comp
    end
    
    Comp --> M1
    Comp --> M2
    Comp --> M3
    
    subgraph "The 3 Pillars of Evaluation"
        M1[SAS<br>Did it mean the same thing?]
        M2[ROUGE-L<br>Did it use the same terminology?]
        M3[LLM Judge<br>Is the fact correct?]
    end
```
