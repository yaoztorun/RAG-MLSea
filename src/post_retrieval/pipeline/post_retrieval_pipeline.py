from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.post_retrieval.pipeline.context_builder import (
    DEFAULT_MIN_RETRIEVAL_SCORE,
    UNANSWERABLE_RESPONSE,
    build_context_payload,
    post_retrieval_pipeline,
    rerank_candidates,
)
from src.post_retrieval.pipeline.data_loading import (
    DEFAULT_CANONICAL_PAPERS_PATH,
    DEFAULT_REPRESENTATIONS_DIR,
    build_paper_id_lookup,
    build_representation_lookup,
    load_canonical_paper_records,
    load_representation_records,
)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def build_offline_lookups(
    *,
    canonical_records_path: str = str(DEFAULT_CANONICAL_PAPERS_PATH),
    representation_type: Optional[str] = None,
    representations_dir: str = str(DEFAULT_REPRESENTATIONS_DIR),
) -> Dict[str, Dict[str, Any]]:
    paper_lookup = build_paper_id_lookup(load_canonical_paper_records(canonical_records_path))
    representation_lookup = None
    if representation_type:
        representation_lookup = build_representation_lookup(
            load_representation_records(representation_type, representations_dir=representations_dir)
        )
    return {
        "paper_lookup": paper_lookup,
        "representation_lookup": representation_lookup,
    }


__all__ = [
    "DEFAULT_CROSS_ENCODER_MODEL",
    "DEFAULT_MIN_RETRIEVAL_SCORE",
    "UNANSWERABLE_RESPONSE",
    "build_context_payload",
    "build_offline_lookups",
    "load_cross_encoder",
    "post_retrieval_pipeline",
    "rerank_candidates",
]
