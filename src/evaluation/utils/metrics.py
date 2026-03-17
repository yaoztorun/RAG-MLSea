from typing import List


def hit_at_k(ranked_ids: List[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: List[str], gold_id: str) -> float:
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            return 1.0 / idx
    return 0.0