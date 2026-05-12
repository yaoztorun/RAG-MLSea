from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.pre_retrieval.shared.utils import truncate_text, unique_preserve_order


LIST_FIELDS = ["tasks", "datasets", "methods", "metrics"]


def _render_list(values: Iterable[str], item_limit: int, value_limit: int) -> str:
    cleaned = [truncate_text(value, value_limit) for value in unique_preserve_order(values)]
    return ", ".join(cleaned[:item_limit])


def build_enriched_paper_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    item_limit = int(config.get("list_item_limit", 5))
    value_limit = int(config.get("list_value_max_characters", 120))
    author_limit = int(config.get("author_limit", 6))
    implementation_limit = int(config.get("implementation_limit", 3))
    parts: List[str] = []

    title = truncate_text(record.get("title"), int(config.get("title_max_characters", 512)))
    abstract = truncate_text(record.get("abstract"), int(config.get("abstract_max_characters", 900)))
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    for field_name in LIST_FIELDS:
        rendered = _render_list(record.get(field_name, []), item_limit, value_limit)
        if rendered:
            parts.append(f"{field_name.replace('_', ' ').title()}: {rendered}")

    authors = _render_list(record.get("authors", []), author_limit, value_limit)
    implementations = _render_list(record.get("implementations", []), implementation_limit, value_limit)
    if authors:
        parts.append(f"Authors: {authors}")
    if implementations:
        parts.append(f"Implementations: {implementations}")

    return truncate_text("\n".join(parts), int(config.get("max_characters", 2200)))
