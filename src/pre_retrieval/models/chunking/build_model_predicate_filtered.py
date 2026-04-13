from __future__ import annotations

from typing import Any, Dict, List

from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text


MODEL_PREDICATE_WHITELIST = [
    "http://purl.org/dc/terms/title",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "https://schema.org/description",
    "http://purl.org/dc/terms/issued",
    "https://schema.org/datePublished",
    "http://www.w3.org/ns/dcat#keyword",
    "http://w3id.org/mlso/hasTaskType",
    "http://w3id.org/mlso/usesDataset",
    "http://w3id.org/mlso/hasRelatedPaper",
    "http://w3id.org/mlso/hasRelatedImplementation",
    "http://w3id.org/mlso/hasEvaluation",
    "http://w3id.org/mlso/hasRun",
]


def build_model_predicate_filtered_text(record: Dict[str, Any], config: Dict[str, Any]) -> str | None:
    """Build predicate-filtered model representation.

    Drops models with NO meaningful predicate information beyond a bare label.
    """
    max_characters = int(config.get("max_characters", 1800))
    title_max = int(config.get("title_max_characters", 512))
    description_max = int(config.get("description_max_characters", 500))
    list_item_limit = int(config.get("list_item_limit", 5))
    list_value_max = int(config.get("list_value_max_characters", 100))

    keywords: List[str] = record.get("keywords") or []
    tasks: List[str] = record.get("tasks") or []
    datasets: List[str] = record.get("datasets") or []
    related_papers: List[str] = record.get("related_papers") or []
    related_impls: List[str] = record.get("related_implementations") or []
    runs: List[str] = record.get("runs") or []
    metrics_list: List[str] = record.get("metrics") or []

    if not (keywords or tasks or datasets or related_papers or related_impls or runs or metrics_list):
        return None  # DROP model completely

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

    if keywords:
        items = [truncate_text(keyword, list_value_max) for keyword in keywords[:list_item_limit]]
        parts.append(f"Keywords: {', '.join(items)}")

    if tasks:
        items = [truncate_text(task, list_value_max) for task in tasks[:list_item_limit]]
        parts.append(f"Tasks: {', '.join(items)}")

    if datasets:
        items = [truncate_text(ds, list_value_max) for ds in datasets[:list_item_limit]]
        parts.append(f"Datasets: {', '.join(items)}")

    if related_papers:
        items = [truncate_text(p, list_value_max) for p in related_papers[:list_item_limit]]
        parts.append(f"Related Papers: {', '.join(items)}")

    if related_impls:
        items = [truncate_text(i, list_value_max) for i in related_impls[:list_item_limit]]
        parts.append(f"Implementations: {', '.join(items)}")

    if metrics_list:
        items = [truncate_text(m, list_value_max) for m in metrics_list[:list_item_limit]]
        parts.append(f"Metrics: {', '.join(items)}")

    text = "\n".join(parts)
    return truncate_text(text, max_characters)
