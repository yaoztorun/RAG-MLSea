from __future__ import annotations

from typing import Any, Dict, List

from src.pre_retrieval.utils import normalize_whitespace, truncate_text


DATASET_PREDICATE_WHITELIST = [
    "http://purl.org/dc/terms/title",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "https://schema.org/description",
    "http://purl.org/dc/terms/issued",
    "https://schema.org/datePublished",
    "http://www.w3.org/ns/dcat#keyword",
    "http://w3id.org/mlso/hasTaskType",
    "http://w3id.org/mlso/hasRelatedPaper",
    "http://w3id.org/mlso/hasRelatedImplementation",
]


def build_dataset_predicate_filtered_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    max_characters = int(config.get("max_characters", 1800))
    title_max = int(config.get("title_max_characters", 512))
    description_max = int(config.get("description_max_characters", 500))
    list_item_limit = int(config.get("list_item_limit", 5))
    list_value_max = int(config.get("list_value_max_characters", 100))

    parts: List[str] = []

    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if label:
        parts.append(f"Dataset: {truncate_text(label, title_max)}")

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

    text = "\n".join(parts)
    return truncate_text(text, max_characters)
