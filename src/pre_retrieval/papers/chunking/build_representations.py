from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from src.pre_retrieval.papers.chunking.build_abstract_only_chunks import build_abstract_only_text
from src.pre_retrieval.papers.chunking.build_enriched_paper_chunks import build_enriched_paper_text
from src.pre_retrieval.papers.chunking.build_one_hop_paper_chunks import build_one_hop_paper_text
from src.pre_retrieval.papers.chunking.build_predicate_filtered_chunks import build_predicate_filtered_text
from src.pre_retrieval.papers.chunking.build_title_abstract_chunks import build_title_abstract_text
from src.pre_retrieval.papers.chunking.build_title_only_chunks import build_title_only_text
from src.pre_retrieval.shared.utils import approx_token_count, build_item_id, compute_distribution_stats, load_jsonl, normalize_identifier, save_json, save_jsonl


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


def _make_paper_fallback_record(record: Dict[str, Any], representation_type: str) -> Optional[Dict[str, Any]]:
    """Build a minimal fallback record for a gold target paper that the normal builder skipped."""
    fallback_text = (
        build_title_only_text(record, {"max_characters": 512})
        or str(record.get("title") or record.get("paper_id", "")).strip()
    )
    if not fallback_text:
        return None
    return {
        "item_id": build_item_id(representation_type, record["paper_id"]),
        "paper_id": record["paper_id"],
        "paper_uri": record.get("paper_uri", record["paper_id"]),
        "title": record.get("title"),
        "representation_type": representation_type,
        "source_text": fallback_text,
        "text_length_chars": len(fallback_text),
        "text_length_tokens_approx": approx_token_count(fallback_text),
        "gold_target_fallback": True,
    }


def build_representations(
    records_path: Path,
    output_dir: Path,
    representation_types: Iterable[str],
    representation_config_map: Dict[str, Dict[str, Any]],
    limit: Optional[int] = None,
    gold_target_ids: Optional[Set[str]] = None,
) -> Dict[str, int]:
    records = load_jsonl(records_path)
    if limit is not None:
        records = records[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for representation_type in representation_types:
        config = representation_config_map.get(representation_type, {})
        built_records: List[Dict[str, Any]] = []
        gold_naturally = 0
        gold_fallback = 0
        gold_missing = 0

        for record in records:
            pid = normalize_identifier(str(record.get("paper_id", "")))
            built = build_representation_record(record, representation_type, config)
            if built is not None:
                built_records.append(built)
                if gold_target_ids and pid in gold_target_ids:
                    gold_naturally += 1
            elif gold_target_ids and pid in gold_target_ids:
                fallback = _make_paper_fallback_record(record, representation_type)
                if fallback is not None:
                    built_records.append(fallback)
                    gold_fallback += 1
                else:
                    gold_missing += 1

        if gold_target_ids:
            total_gold = gold_naturally + gold_fallback
            print(f"[Gold Papers] Included in {representation_type}: {total_gold}/{len(gold_target_ids)}", flush=True)
            if gold_fallback:
                print(f"[Gold Papers] Force-included via fallback in {representation_type}: {gold_fallback}", flush=True)
            if gold_missing:
                print(f"[Gold Papers] Missing from representation {representation_type}: {gold_missing}", flush=True)
            else:
                print(f"[Gold Papers] Missing from representation {representation_type}: 0", flush=True)

        save_jsonl(built_records, output_dir / f"{representation_type}.jsonl")
        save_json(build_representation_stats(built_records, representation_type), output_dir / f"{representation_type}_stats.json")
        counts[representation_type] = len(built_records)
    return counts
