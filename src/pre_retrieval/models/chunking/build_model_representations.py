from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.pre_retrieval.models.chunking.build_model_title_only import build_model_title_only_text
from src.pre_retrieval.models.chunking.build_model_metadata import build_model_metadata_text
from src.pre_retrieval.models.chunking.build_model_predicate_filtered import build_model_predicate_filtered_text
from src.pre_retrieval.models.chunking.build_model_enriched_metadata import build_model_enriched_metadata_text
from src.pre_retrieval.shared.utils import approx_token_count, build_item_id, compute_distribution_stats, load_jsonl, save_json, save_jsonl


SUPPORTED_MODEL_REPRESENTATIONS = [
    "model_title_only",
    "model_metadata",
    "model_predicate_filtered",
    "model_enriched_metadata",
]

MODEL_BUILDER_MAP: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], str]] = {
    "model_title_only": build_model_title_only_text,
    "model_metadata": build_model_metadata_text,
    "model_predicate_filtered": build_model_predicate_filtered_text,
    "model_enriched_metadata": build_model_enriched_metadata_text,
}


def build_model_representation_record(record: Dict[str, Any], representation_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_text = MODEL_BUILDER_MAP[representation_type](record, config)
    if not source_text:
        return None
    return {
        "item_id": build_item_id(representation_type, record["model_id"]),
        "model_id": record["model_id"],
        "model_uri": record.get("model_uri", record["model_id"]),
        "title": record.get("label") or record.get("title"),
        "representation_type": representation_type,
        "source_text": source_text,
        "text_length_chars": len(source_text),
        "text_length_tokens_approx": approx_token_count(source_text),
    }


def build_model_representation_stats(records: List[Dict[str, Any]], representation_type: str) -> Dict[str, Any]:
    char_lengths = [record["text_length_chars"] for record in records]
    token_lengths = [record["text_length_tokens_approx"] for record in records]
    return {
        "representation_type": representation_type,
        "record_count": len(records),
        "chars": compute_distribution_stats(char_lengths),
        "tokens_approx": compute_distribution_stats(token_lengths),
    }


def build_model_representations(
    records_path: Path,
    output_dir: Path,
    representation_types: Iterable[str],
    representation_config_map: Dict[str, Dict[str, Any]],
    limit: Optional[int] = None,
) -> Dict[str, int]:
    records = load_jsonl(records_path)
    if limit is not None:
        records = records[:limit]

    # Defensive dedup: keep the first record per model_id
    seen_ids: set[str] = set()
    unique_records: List[Dict[str, Any]] = []
    for record in records:
        mid = record.get("model_id", "")
        if mid not in seen_ids:
            seen_ids.add(mid)
            unique_records.append(record)
    if len(unique_records) < len(records):
        print(f"[build_model_representations] deduplicated {len(records) - len(unique_records)} duplicate model_id(s) from input records.", flush=True)
    records = unique_records

    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for representation_type in representation_types:
        config = representation_config_map.get(representation_type, {})
        built_records: List[Dict[str, Any]] = []
        for record in records:
            built = build_model_representation_record(record, representation_type, config)
            if built is not None:
                built_records.append(built)

        save_jsonl(built_records, output_dir / f"{representation_type}.jsonl")
        save_json(build_model_representation_stats(built_records, representation_type), output_dir / f"{representation_type}_stats.json")
        counts[representation_type] = len(built_records)
    return counts
