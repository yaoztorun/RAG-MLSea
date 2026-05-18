from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.post_retrieval.pipeline.context_builder import UNANSWERABLE_RESPONSE
from src.post_retrieval.pipeline.post_retrieval_pipeline import build_context_payload

DEFAULT_GENERATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_generation_model(
    model_id: str = DEFAULT_GENERATION_MODEL,
    *,
    device: Optional[str] = None,
    torch_dtype: str = "auto",
) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs: Dict[str, Any] = {}
    if torch_dtype == "auto":
        model_kwargs["dtype"] = "auto"
    elif hasattr(torch, torch_dtype):
        model_kwargs["dtype"] = getattr(torch, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(runtime_device)
    return model, tokenizer, runtime_device


def generate_rag_answer(
    question: str,
    context: str,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    if context.strip() == UNANSWERABLE_RESPONSE:
        return UNANSWERABLE_RESPONSE

    system_prompt = (
        "You are an expert machine learning assistant operating on an offline MLSea RAG pipeline. "
        "The provided context was assembled from offline canonical paper records and retrieval representations. "
        "Answer the user's question using only that context. If the context is insufficient, say that the "
        "question is unanswerable from the available offline MLSea context."
    )
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": question},
    ]
    prompt_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(response, skip_special_tokens=True).strip()


def generate_baseline_answer(
    question: str,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 128, # Increased to prevent cut-offs
    temperature: float = 0.1,
) -> str:
    """Generates an answer to the question WITHOUT context (baseline)."""
    messages = [
        {
            "role": "system", 
            "content": "You are a concise assistant. Answer in 1-2 complete sentences. Do not use bullet points or lists."
        },
        {"role": "user", "content": question},
    ]
    prompt_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(response, skip_special_tokens=True).strip()
    return text.split("<|")[0].strip()


def judge_rag_answer(
    ground_truth: str,
    generated_answer: str,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[int, str]:
    # Few-shot CoT Prompt with explicit STOP marker
    prompt = (
        f"Goal: Is the Answer factually correct relative to the Ground Truth?\n"
        f"Rules: Explain reasoning, then output [[1]] or [[0]]. End with '###'.\n\n"
        f"Example 1:\nGT: Paris is in France.\nAns: Paris is French.\nReasoning: Both mention Paris. [[1]] ###\n\n"
        f"Example 2:\nGT: Paris is in France.\nAns: Paris is in Italy.\nReasoning: Italy is not France. [[0]] ###\n\n"
        f"Current Task:\nGT: {ground_truth}\nAns: {generated_answer}\nReasoning:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the new tokens
    response = outputs[0][inputs["input_ids"].shape[-1] :]
    text_response = tokenizer.decode(response, skip_special_tokens=True).strip()

    # Split at the STOP marker to prevent rambling into examples
    text_response = text_response.split("###")[0].strip()

    # Robust parsing: check for [[1]] first, then fall back to standalone 0/1
    import re
    match = re.search(r"\[\[([01])\]\]", text_response)
    if not match:
        match = re.search(r"(?:^|\s|:)([01])(?:\s|$|\.)", text_response)
    
    score = int(match.group(1)) if match else 0
    return score, text_response


def generate_answer_from_retrieval(
    question: str,
    retrieved_results: Iterable[Dict[str, Any]],
    paper_lookup: Dict[str, Dict[str, Any]],
    *,
    representation_lookup: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None,
    cross_encoder: Any = None,
    use_cross_encoder: bool = True,
    min_retrieval_score: Optional[float] = 0.20,
    top_k: int = 3,
    model: Any = None,
    tokenizer: Any = None,
    device: Optional[str] = None,
    model_id: str = DEFAULT_GENERATION_MODEL,
) -> Dict[str, Any]:
    payload = build_context_payload(
        question,
        retrieved_results,
        paper_lookup,
        representation_lookup=representation_lookup,
        cross_encoder=cross_encoder,
        use_cross_encoder=use_cross_encoder,
        min_retrieval_score=min_retrieval_score,
        top_k=top_k,
    )
    if payload["context"] == UNANSWERABLE_RESPONSE:
        payload["answer"] = UNANSWERABLE_RESPONSE
        return payload

    runtime_model = model
    runtime_tokenizer = tokenizer
    runtime_device = device
    if runtime_model is None or runtime_tokenizer is None or runtime_device is None:
        runtime_model, runtime_tokenizer, runtime_device = load_generation_model(model_id=model_id, device=device)

    payload["answer"] = generate_rag_answer(
        question,
        payload["context"],
        model=runtime_model,
        tokenizer=runtime_tokenizer,
        device=runtime_device,
    )
    return payload
