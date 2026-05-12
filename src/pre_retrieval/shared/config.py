from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pre_retrieval_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "embedder_type": "sentence_transformer",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "corpus_subset": {
        "enabled": True,
        "max_papers": 200000,
        "include_gold_targets": True,
    },
    "vector_store": {
        "provider": "chroma",
        "chroma_mode": "http",
        "chroma_host": "localhost",
        "chroma_port": 8000,
        "persist_directory": "data/intermediate/chroma",
    },
    "evaluation": {
        "questions_path": "data/questions/ml_questions_dataset.json",
        "output_dir": "data/retrieval_results",
        "top_k": [1, 5, 10],
        "abstention_score_threshold": None,
        "representation_order": [
            "title_only",
            "abstract_only",
            "title_abstract",
            "predicate_filtered",
            "enriched_metadata",
            "one_hop",
        ],
        "dataset_representation_order": [
            "dataset_title_only",
            "dataset_metadata",
            "dataset_predicate_filtered",
            "dataset_enriched_metadata",
        ],
        "model_representation_order": [
            "model_title_only",
            "model_metadata",
            "model_predicate_filtered",
            "model_enriched_metadata",
        ],
    },
    "representations": {
        "title_only": {"max_characters": 512},
        "abstract_only": {"max_characters": 1600},
        "title_abstract": {
            "title_max_characters": 512,
            "abstract_max_characters": 1400,
            "max_characters": 1800,
        },
        "enriched_metadata": {
            "title_max_characters": 512,
            "abstract_max_characters": 900,
            "list_item_limit": 5,
            "list_value_max_characters": 120,
            "author_limit": 6,
            "implementation_limit": 3,
            "max_characters": 2200,
        },
        "predicate_filtered": {
            "title_max_characters": 512,
            "abstract_max_characters": 500,
            "list_item_limit": 5,
            "list_value_max_characters": 100,
            "max_characters": 1800,
        },
        "one_hop": {
            "title_max_characters": 512,
            "abstract_max_characters": 700,
            "linked_entity_limit": 12,
            "list_value_max_characters": 100,
            "max_characters": 2200,
        },
    },
    "dataset_representations": {
        "dataset_title_only": {"max_characters": 512},
        "dataset_metadata": {
            "title_max_characters": 512,
            "description_max_characters": 900,
            "list_item_limit": 10,
            "list_value_max_characters": 120,
            "max_characters": 2200,
        },
        "dataset_predicate_filtered": {
            "title_max_characters": 512,
            "description_max_characters": 500,
            "list_item_limit": 5,
            "list_value_max_characters": 100,
            "max_characters": 1800,
        },
        "dataset_enriched_metadata": {
            "title_max_characters": 512,
            "description_max_characters": 600,
            "list_item_limit": 8,
            "list_value_max_characters": 120,
            "related_paper_limit": 6,
            "implementation_limit": 4,
            "linked_entity_limit": 6,
            "max_characters": 2400,
        },
    },
    "model_representations": {
        "model_title_only": {"max_characters": 512},
        "model_metadata": {
            "title_max_characters": 512,
            "list_item_limit": 10,
            "list_value_max_characters": 120,
            "max_characters": 2200,
        },
        "model_predicate_filtered": {
            "title_max_characters": 512,
            "list_item_limit": 5,
            "list_value_max_characters": 100,
            "max_characters": 1800,
        },
        "model_enriched_metadata": {
            "title_max_characters": 512,
            "description_max_characters": 600,
            "list_item_limit": 8,
            "list_value_max_characters": 120,
            "linked_entity_limit": 6,
            "max_characters": 2400,
        },
    },
}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def format_paper_count_suffix(max_papers: int) -> str:
    if max_papers % 1_000 == 0:
        return f"{max_papers // 1_000}k"
    return str(max_papers)


def default_subset_records_path(max_papers: int) -> str:
    return f"data/intermediate/raw_papers/papers_subset_{format_paper_count_suffix(max_papers)}.jsonl"


def default_subset_stats_path() -> str:
    return "data/intermediate/raw_papers/subset_stats.json"


def resolve_records_path(
    config: Dict[str, Any],
    input_path: str | Path | None = None,
    *,
    disable_subset: bool = False,
    max_papers: int | None = None,
) -> Path:
    if input_path is not None:
        return resolve_repo_path(input_path)

    subset_config = config.get("corpus_subset", {})
    subset_enabled = bool(subset_config.get("enabled", False)) and not disable_subset
    if subset_enabled:
        subset_max_papers = int(max_papers or subset_config.get("max_papers", 200000))
        return resolve_repo_path(default_subset_records_path(subset_max_papers))
    return resolve_repo_path("data/intermediate/raw_papers/papers_master.jsonl")


def load_pipeline_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_repo_path(config_path or DEFAULT_CONFIG_PATH)
    loaded: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)

    config = _deep_merge(DEFAULT_CONFIG, loaded)
    vector_store_config = config.get("vector_store", {})
    if "db_path" in vector_store_config and "persist_directory" not in vector_store_config:
        vector_store_config["persist_directory"] = vector_store_config["db_path"]
    config["_config_path"] = str(path)
    return config
