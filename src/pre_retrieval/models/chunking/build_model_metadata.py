from __future__ import annotations

from typing import Any, Dict, List

from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text


def build_model_metadata_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    max_characters = int(config.get("max_characters", 2200))
    title_max = int(config.get("title_max_characters", 512))
    description_max = int(config.get("description_max_characters", 900))
    list_item_limit = int(config.get("list_item_limit", 10))
    list_value_max = int(config.get("list_value_max_characters", 120))

    parts: List[str] = []

    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if label:
        parts.append(f"Model: {truncate_text(label, title_max)}")

    description = normalize_whitespace(record.get("description") or "")
    if description:
        parts.append(f"Description: {truncate_text(description, description_max)}")

    issued_year = normalize_whitespace(record.get("issued_year") or "")
    if issued_year:
        parts.append(f"Year: {issued_year}")

    keywords: List[str] = record.get("keywords") or []
    if keywords:
        items = [truncate_text(keyword, list_value_max) for keyword in keywords[:list_item_limit]]
        parts.append(f"Keywords: {', '.join(items)}")

    tasks: List[str] = record.get("tasks") or []
    if tasks:
        items = [truncate_text(task, list_value_max) for task in tasks[:list_item_limit]]
        parts.append(f"Tasks: {', '.join(items)}")

    datasets: List[str] = record.get("datasets") or []
    if datasets:
        items = [truncate_text(ds, list_value_max) for ds in datasets[:list_item_limit]]
        parts.append(f"Datasets: {', '.join(items)}")

    text = "\n".join(parts)
    return truncate_text(text, max_characters)
