from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

QUESTIONS_PATH = REPO_ROOT / "data" / "questions" / "ml_questions_dataset.json"
RETRIEVAL_OUTPUT_DIR = REPO_ROOT / "data" / "results" / "retrieval"

# Ordered by priority; first existing path wins.
PRE_RETRIEVAL_BASE_PATHS = [
    REPO_ROOT / "data" / "results" / "pre_retrieval_results",
    REPO_ROOT / "data" / "results" / "pre_retrieval",
    REPO_ROOT / "data" / "retrieval_results",
]

ENTITY_TYPE_FOLDERS: dict[str, str] = {
    "paper": "paper_results",
    "dataset": "dataset_results",
    "model": "model_results",
}

BEST_REPRESENTATION: dict[str, str] = {
    "paper": "enriched_metadata",
    "dataset": "dataset_title_only",
    "model": "model_predicate_filtered",
}

RRF_FUSION_GROUPS: dict[str, list[str]] = {
    "paper": ["enriched_metadata", "predicate_filtered", "one_hop"],
    "dataset": ["dataset_title_only", "dataset_enriched_metadata"],
    "model": ["model_predicate_filtered", "model_enriched_metadata"],
}

DEFAULT_TOP_K = 10
RRF_K = 60

# -----------------------------------------------------------------------
# Method registry — thesis-facing names
# -----------------------------------------------------------------------

MAIN_METHODS: list[str] = [
    "pure_semantic_dense",
    "hybrid_type_filtering",
    "hybrid_type_onehop_filtering",
    "hybrid_predicate_aware_filtering",
]

OPTIONAL_METHODS: list[str] = [
    "optional_rrf_fusion",
    "optional_rrf_symbolic",
]

ALL_METHODS: list[str] = MAIN_METHODS + OPTIONAL_METHODS

# Backward-compatibility: old name -> canonical new name
METHOD_ALIASES: dict[str, str] = {
    "dense_baseline": "pure_semantic_dense",
    "type_filtering": "hybrid_type_filtering",
    "predicate_aware_filtering": "hybrid_predicate_aware_filtering",
    "rrf_fusion": "optional_rrf_fusion",
    "rrf_symbolic": "optional_rrf_symbolic",
}

METHOD_GROUPS: dict[str, str] = {
    "pure_semantic_dense": "Pure Semantic",
    "hybrid_type_filtering": "Hybrid",
    "hybrid_type_onehop_filtering": "Hybrid",
    "hybrid_predicate_aware_filtering": "Hybrid",
    "optional_rrf_fusion": "Optional Fusion",
    "optional_rrf_symbolic": "Optional Fusion",
}

INTERPRETATION_LABELS: dict[str, str] = {
    "pure_semantic_dense": "precision baseline",
    "hybrid_type_filtering": "type-control",
    "hybrid_type_onehop_filtering": "graph-neighbourhood hybrid",
    "hybrid_predicate_aware_filtering": "predicate-aware hybrid",
    "optional_rrf_fusion": "optional recall fusion",
    "optional_rrf_symbolic": "optional fusion + symbolic",
}

METRIC_NAMES: list[str] = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG"]
