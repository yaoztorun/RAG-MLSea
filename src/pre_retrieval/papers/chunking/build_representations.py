from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.pre_retrieval.papers.chunking.build_abstract_only_chunks import build_abstract_only_text
from src.pre_retrieval.papers.chunking.build_enriched_paper_chunks import build_enriched_paper_text
from src.pre_retrieval.papers.chunking.build_one_hop_paper_chunks import build_one_hop_paper_text
from src.pre_retrieval.papers.chunking.build_predicate_filtered_chunks import build_predicate_filtered_text
from src.pre_retrieval.papers.chunking.build_title_abstract_chunks import build_title_abstract_text
from src.pre_retrieval.papers.chunking.build_title_only_chunks import build_title_only_text
from src.pre_retrieval.shared.utils import approx_token_count, build_item_id, compute_distribution_stats, load_jsonl, save_json, save_jsonl


SUPPORTED_REPRESENTATIONS = [
    "title_only",
    "abstract_only",
    "title_abstract",
    "enriched_metadata",
    "predicate_filtered",
    "one_hop",
]

BUILDER_MAP: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], str]] = {
    "title_only": build_title_only_text,
    "abstract_only": build_abstract_only_text,
    "title_abstract": build_title_abstract_text,
    "enriched_metadata": build_enriched_paper_text,
    "predicate_filtered": build_predicate_filtered_text,
    "one_hop": build_one_hop_paper_text,
}


def build_representation_record(record: Dict[str, Any], representation_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_text = BUILDER_MAP[representation_type](record, config)
    if not source_text:
        return None
    return {
        "item_id": build_item_id(representation_type, record["paper_id"]),
        "paper_id": record["paper_id"],
        "paper_uri": record.get("paper_uri", record["paper_id"]),
        "title": record.get("title"),
        "representation_type": representation_type,
        "source_text": source_text,
        "text_length_chars": len(source_text),
        "text_length_tokens_approx": approx_token_count(source_text),
    }


def build_representation_stats(records: List[Dict[str, Any]], representation_type: str) -> Dict[str, Any]:
    char_lengths = [record["text_length_chars"] for record in records]
    token_lengths = [record["text_length_tokens_approx"] for record in records]
    return {
        "representation_type": representation_type,
        "record_count": len(records),
        "chars": compute_distribution_stats(char_lengths),
        "tokens_approx": compute_distribution_stats(token_lengths),
    }


def build_representations(
    records_path: Path,
    output_dir: Path,
    representation_types: Iterable[str],
    representation_config_map: Dict[str, Dict[str, Any]],
    limit: Optional[int] = None,
) -> Dict[str, int]:
    records = load_jsonl(records_path)
    if limit is not None:
        records = records[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for representation_type in representation_types:
        config = representation_config_map.get(representation_type, {})
        built_records: List[Dict[str, Any]] = []
        for record in records:
            built = build_representation_record(record, representation_type, config)
            if built is not None:
                built_records.append(built)

        save_jsonl(built_records, output_dir / f"{representation_type}.jsonl")
        save_json(build_representation_stats(built_records, representation_type), output_dir / f"{representation_type}_stats.json")
        counts[representation_type] = len(built_records)
    return counts
