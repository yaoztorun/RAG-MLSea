import torch
from src.post_retrieval.generation.llama_generation import load_generation_model, judge_rag_answer

def test_judge():
    print("Loading model for isolated judge test...")
    model, tokenizer, device = load_generation_model()
    
    # Test Case 1: Obvious Match
    gt = "The capital of France is Paris."
    ans = "Paris is the capital of France."
    print(f"\nTest 1 (Should be 1):")
    score, raw = judge_rag_answer(gt, ans, model=model, tokenizer=tokenizer, device=device)
    print(f"Score: {score}")
    print(f"Raw: {raw}")

    # Test Case 2: Obvious Mismatch
    gt = "The capital of France is Paris."
    ans = "London is the capital of France."
    print(f"\nTest 2 (Should be 0):")
    score, raw = judge_rag_answer(gt, ans, model=model, tokenizer=tokenizer, device=device)
    print(f"Score: {score}")
    print(f"Raw: {raw}")

if __name__ == "__main__":
    test_judge()
