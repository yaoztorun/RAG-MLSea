import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Import the exact post-retrieval pipeline we built earlier
from post_retrieval_pipeline import (
    post_retrieval_pipeline, run_sparql, build_paper_chunks, ENDPOINT, QUERY_ALL_PAPERS
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# Setup SLURM / GPU environment variables if scaling on VSC
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_rag_answer(question: str, context: str, model, tokenizer):
    """
    Feeds the strict MLSea Knowledge Graph context and the user query
    directly into LLaMA-3 natively using HuggingFace's chat template infrastructure.
    """
    
    # 1. We strictly instruct LLaMA to act as a grounded RAG agent
    system_prompt = (
        "You are an expert Machine Learning assistant executing a Retrieval-Augmented Generation task. "
        "I will provide a strictly verified context block sourced directly from the MLSea Knowledge Graph. "
        "You MUST answer the user's question relying strictly on this context. "
        "If the context explicitly states 'The question is unanswerable', repeat that. "
        "Do not hallucinate external information or papers that are not securely listed in the context."
    )
    
    # 2. We use the official LLaMA v3 chat instruction JSON formatting
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": question},
    ]

    # Convert the generic messages dict into LLaMA's specific token format (<|start_header_id|>, etc.)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 3. Generation Configurations (Prevent hallucinations via low temperature)
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.1,  # Keep it almost deterministic (greedy search) to heavily anchor the context
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return just the generated answer, stripping out the massive prompt context
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def main():
    # Standard open-source foundation model for RAG benchmarking
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"1. Loading Generation LLM ({model_id}) on {device.upper()}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # We load in bfloat16 (16-bit) to fit an 8B model cleanly onto VSC SLURM consumer/cluster GPUs
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    except Exception as e:
        print(f"Warning: Failed to load LLaMA successfully. (Have you run 'huggingface-cli login' to access weights via your token?)\nException Error: {e}")
        return

    # 2. Load the Post-Retrieval Models
    print("2. Loading SBERT and Cross-Encoder from HuggingFace Hub...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 3. Fetch from GraphDB (VSC cluster jobs will normally do this via REST over the local intranet network)
    print(f"3. Connecting to {ENDPOINT} to extract raw semantic graph chunks...")
    try:
        rows = run_sparql(ENDPOINT, QUERY_ALL_PAPERS)
        chunks = build_paper_chunks(rows)
        chunk_embeddings = bi_encoder.encode([c["text"] for c in chunks])
    except Exception as e:
        print(f"GraphDB Connection failed: {e}")
        return

    # 4. Our Test User Question
    test_question = "Which papers discuss the concept of contrastive learning?"

    # ==========================
    # FULL RAG PIPELINE INTEGRATION
    # ==========================
    print(f"\n{'='*40}")
    print("STARTING END-TO-END RAG PIPELINE")
    print(f"{'='*40}")
    start_time = time.time()
    
    # -> A) Run the exact Post-Retrieval Pipeline we just built
    print("-> Reranking and Formatting graph data...")
    final_context = post_retrieval_pipeline(test_question, chunks, chunk_embeddings, bi_encoder, cross_encoder)
    
    # -> B) Run the Local Generation Phase with LLaMA
    print("-> Reading Graph Context and Generating intelligent answer dynamically with LLaMA 3...")
    answer = generate_rag_answer(test_question, final_context, model, tokenizer)
    
    end_time = time.time()
    
    print(f"\n================ FINAL NATIVE LLaMA RAG ANSWER ================\n")
    print(answer)
    print(f"\n===============================================================\n")
    print(f"Total VSC Node Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
