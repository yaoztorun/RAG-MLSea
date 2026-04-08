from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.post_retrieval.pipeline.data_loading import resolve_representation_text
from src.pre_retrieval.utils import normalize_identifier, truncate_text

DEFAULT_MIN_RETRIEVAL_SCORE = 0.20
MAX_ABSTRACT_CONTEXT_CHARS = 1200
MAX_REPRESENTATION_CONTEXT_CHARS = 1400
UNANSWERABLE_RESPONSE = "The question is unanswerable from the available offline MLSea context."


def _stringify_score(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_list(values: Iterable[str], fallback: str = "Unknown") -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(cleaned) if cleaned else fallback


def build_candidate_context(
    result: Dict[str, Any],
    canonical_record: Dict[str, Any],
    representation_text: str,
) -> str:
    title = canonical_record.get("title") or result.get("title") or "Unknown"
    year = canonical_record.get("year") or "Unknown"
    abstract = truncate_text(canonical_record.get("abstract", ""), MAX_ABSTRACT_CONTEXT_CHARS)
    representation_excerpt = truncate_text(representation_text, MAX_REPRESENTATION_CONTEXT_CHARS)

    lines = [
        f"Title: {title}",
        f"Publication Year: {year}",
        f"Authors: {_format_list(canonical_record.get('authors', []))}",
        f"Tasks: {_format_list(canonical_record.get('tasks', []), fallback='None')}",
        f"Keywords: {_format_list(canonical_record.get('keywords', []), fallback='None')}",
        f"Datasets: {_format_list(canonical_record.get('datasets', []), fallback='None')}",
        f"Methods: {_format_list(canonical_record.get('methods', []), fallback='None')}",
        f"Metrics: {_format_list(canonical_record.get('metrics', []), fallback='None')}",
        f"Implementations: {_format_list(canonical_record.get('implementations', []), fallback='None')}",
    ]
    if abstract:
        lines.append(f"Abstract: {abstract}")
    if representation_excerpt:
        lines.append(f"Retrieved Representation: {representation_excerpt}")
    return "\n".join(lines)


def build_candidate_payload(
    result: Dict[str, Any],
    paper_lookup: Dict[str, Dict[str, Any]],
    representation_lookup: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    paper_id = normalize_identifier(result.get("paper_id", ""))
    canonical_record = paper_lookup.get(paper_id, {})
    representation_text = resolve_representation_text(result, representation_lookup)
    payload = dict(result)
    payload["paper_id"] = paper_id
    payload["entity_iri"] = canonical_record.get("paper_uri", paper_id)
    payload["canonical_record"] = canonical_record
    payload["representation_text"] = representation_text
    payload["retrieval_score"] = _stringify_score(result.get("score"))
    payload["context_text"] = build_candidate_context(payload, canonical_record, representation_text)
    return payload


def filter_candidates(
    candidates: Iterable[Dict[str, Any]],
    min_retrieval_score: Optional[float] = DEFAULT_MIN_RETRIEVAL_SCORE,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        retrieval_score = _stringify_score(candidate.get("retrieval_score"))
        if min_retrieval_score is not None and retrieval_score is not None and retrieval_score < min_retrieval_score:
            continue
        filtered.append(candidate)
    return filtered


def rerank_candidates(
    question: str,
    candidates: Iterable[Dict[str, Any]],
    *,
    cross_encoder: Any = None,
    use_cross_encoder: bool = True,
) -> List[Dict[str, Any]]:
    ranked = [dict(candidate) for candidate in candidates]
    if use_cross_encoder and cross_encoder is not None and ranked:
        cross_inputs = [[question, candidate["context_text"]] for candidate in ranked]
        scores = cross_encoder.predict(cross_inputs)
        for candidate, score in zip(ranked, scores):
            candidate["cross_score"] = float(score)
        ranked.sort(key=lambda candidate: candidate.get("cross_score", float("-inf")), reverse=True)
        return ranked

    ranked.sort(key=lambda candidate: candidate.get("retrieval_score") or float("-inf"), reverse=True)
    return ranked


def format_context_block(selected_results: Iterable[Dict[str, Any]]) -> str:
    lines = ["<CONTEXT>"]
    for index, candidate in enumerate(selected_results, start=1):
        lines.append(f"--- Result {index} ---")
        lines.append(f"Entity IRI: {candidate.get('entity_iri') or candidate.get('paper_id', '')}")
        retrieval_score = _stringify_score(candidate.get("retrieval_score"))
        if retrieval_score is not None:
            lines.append(f"Retrieval Score: {retrieval_score:.4f}")
        cross_score = _stringify_score(candidate.get("cross_score"))
        if cross_score is not None:
            lines.append(f"Cross-Encoder Score: {cross_score:.4f}")
        lines.append(candidate["context_text"])
        lines.append("")
    lines.append("</CONTEXT>")
    return "\n".join(lines)


def build_context_payload(
    question: str,
    retrieved_results: Iterable[Dict[str, Any]],
    paper_lookup: Dict[str, Dict[str, Any]],
    *,
    representation_lookup: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None,
    cross_encoder: Any = None,
    use_cross_encoder: bool = True,
    min_retrieval_score: Optional[float] = DEFAULT_MIN_RETRIEVAL_SCORE,
    top_k: int = 3,
) -> Dict[str, Any]:
    candidates = [
        build_candidate_payload(result, paper_lookup, representation_lookup)
        for result in retrieved_results
        if result.get("paper_id")
    ]
    filtered_results = filter_candidates(candidates, min_retrieval_score=min_retrieval_score)
    if not filtered_results:
        return {
            "question": question,
            "context": UNANSWERABLE_RESPONSE,
            "candidates": candidates,
            "filtered_results": [],
            "ranked_results": [],
            "selected_results": [],
        }

    ranked_results = rerank_candidates(
        question,
        filtered_results,
        cross_encoder=cross_encoder,
        use_cross_encoder=use_cross_encoder,
    )
    selected_results = ranked_results[:top_k]
    return {
        "question": question,
        "context": format_context_block(selected_results),
        "candidates": candidates,
        "filtered_results": filtered_results,
        "ranked_results": ranked_results,
        "selected_results": selected_results,
    }


def post_retrieval_pipeline(
    question: str,
    retrieved_results: Iterable[Dict[str, Any]],
    paper_lookup: Dict[str, Dict[str, Any]],
    *,
    representation_lookup: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None,
    cross_encoder: Any = None,
    use_cross_encoder: bool = True,
    min_retrieval_score: Optional[float] = DEFAULT_MIN_RETRIEVAL_SCORE,
    top_k: int = 3,
) -> str:
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
    return str(payload["context"])
