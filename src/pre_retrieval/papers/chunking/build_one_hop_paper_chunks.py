from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.pre_retrieval.shared.utils import truncate_text, unique_preserve_order


def _render_list(values: Iterable[str], item_limit: int, value_limit: int) -> str:
    cleaned = [truncate_text(value, value_limit) for value in unique_preserve_order(values)]
    return ", ".join(cleaned[:item_limit])


def build_one_hop_paper_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    linked_limit = int(config.get("linked_entity_limit", 12))
    value_limit = int(config.get("list_value_max_characters", 100))
    parts: List[str] = []

    title = truncate_text(record.get("title"), int(config.get("title_max_characters", 512)))
    abstract = truncate_text(record.get("abstract"), int(config.get("abstract_max_characters", 700)))
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    grouped: Dict[str, List[str]] = {}
    for entity in record.get("linked_entities", [])[:linked_limit]:
        category = str(entity.get("category", "linked_entity")).replace("_", " ").title()
        grouped.setdefault(category, []).append(str(entity.get("object_label", "")))

    for category, values in grouped.items():
        rendered = _render_list(values, linked_limit, value_limit)
        if rendered:
            parts.append(f"{category}: {rendered}")

    return truncate_text("\n".join(parts), int(config.get("max_characters", 2200)))
