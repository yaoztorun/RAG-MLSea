from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pre_retrieval.io_utils import (
    build_document_id,
    load_jsonl,
    save_jsonl,
    truncate_text,
    unique_preserve_order,
)


SUPPORTED_REPRESENTATIONS = [
    "title_only",
    "abstract_only",
    "title_abstract",
    "enriched_metadata",
    "one_hop",
]
LIST_PRIORITY = ["tasks", "datasets", "methods", "metrics", "keywords"]


def _render_list(values: Iterable[str], limit: int, value_limit: int) -> str:
    cleaned = [truncate_text(value, value_limit) for value in unique_preserve_order(values)]
    return ", ".join(cleaned[:limit])


def _append(parts: List[str], label: str, value: str) -> None:
    if value:
        parts.append(f"{label}: {value}")


def build_representation_text(
    record: Dict[str, Any],
    representation_type: str,
    representation_config: Dict[str, Any],
) -> str:
    title = truncate_text(record.get("title", ""), representation_config.get("title_max_characters", representation_config.get("max_characters", 512)))
    abstract = truncate_text(record.get("abstract", ""), representation_config.get("abstract_max_characters", representation_config.get("max_characters", 1600)))
    max_characters = int(representation_config.get("max_characters", 2200))

    if representation_type == "title_only":
        return truncate_text(title, max_characters)

    if representation_type == "abstract_only":
        return truncate_text(abstract, max_characters)

    if representation_type == "title_abstract":
        parts: List[str] = []
        _append(parts, "Title", title)
        _append(parts, "Abstract", abstract)
        return truncate_text("\n".join(parts), max_characters)

    if representation_type == "enriched_metadata":
        list_limit = int(representation_config.get("list_item_limit", 5))
        value_limit = int(representation_config.get("list_value_max_characters", 120))
        author_limit = int(representation_config.get("author_limit", list_limit))
        implementation_limit = int(representation_config.get("implementation_limit", 3))

        parts = []
        _append(parts, "Title", title)
        if abstract:
            _append(parts, "Abstract", abstract)
        for field_name in LIST_PRIORITY:
            values = record.get(field_name, [])
            rendered = _render_list(values, list_limit, value_limit)
            if rendered:
                _append(parts, field_name.replace("_", " ").title(), rendered)
        authors = _render_list(record.get("authors", []), author_limit, value_limit)
        implementations = _render_list(record.get("implementations", []), implementation_limit, value_limit)
        _append(parts, "Authors", authors)
        _append(parts, "Implementations", implementations)
        if record.get("year"):
            _append(parts, "Year", str(record["year"]))
        return truncate_text("\n".join(parts), max_characters)

    if representation_type == "one_hop":
        linked_limit = int(representation_config.get("linked_entity_limit", 12))
        parts = []
        _append(parts, "Title", title)
        if abstract:
            _append(parts, "Abstract", abstract)

        linked_entities = record.get("linked_entities", [])[:linked_limit]
        grouped: Dict[str, List[str]] = {}
        for entity in linked_entities:
            bucket = entity.get("category", "linked_entity").replace("_", " ").title()
            grouped.setdefault(bucket, []).append(entity.get("object_label", ""))

        for bucket, values in grouped.items():
            rendered = _render_list(values, linked_limit, 100)
            _append(parts, bucket, rendered)

        if not grouped:
            for field_name in ["tasks", "datasets", "methods", "metrics", "keywords", "implementations"]:
                rendered = _render_list(record.get(field_name, []), 4, 100)
                _append(parts, field_name.replace("_", " ").title(), rendered)

        return truncate_text("\n".join(parts), max_characters)

    raise ValueError(f"Unsupported representation type: {representation_type}")


def build_representation_record(
    record: Dict[str, Any],
    representation_type: str,
    representation_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_text = build_representation_text(record, representation_type, representation_config)
    if not source_text:
        return None

    return {
        "id": build_document_id(representation_type, record["paper_id"]),
        "paper_id": record["paper_id"],
        "title": record.get("title", ""),
        "representation_type": representation_type,
        "source_text": source_text,
        "text_length": len(source_text),
        "year": record.get("year", ""),
        "field_stats": record.get("field_stats", {}),
    }


def build_representations(
    records_path: Path,
    output_dir: Path,
    representation_types: Iterable[str],
    representation_config_map: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    records = load_jsonl(records_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for representation_type in representation_types:
        config = representation_config_map.get(representation_type, {})
        built = []
        for record in records:
            representation_record = build_representation_record(record, representation_type, config)
            if representation_record is not None:
                built.append(representation_record)

        save_jsonl(built, output_dir / f"{representation_type}.jsonl")
        counts[representation_type] = len(built)

    return counts
