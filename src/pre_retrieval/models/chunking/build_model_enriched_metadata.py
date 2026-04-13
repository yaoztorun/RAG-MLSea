from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text, unique_preserve_order


def _render_list(values: Iterable[str], item_limit: int, value_limit: int) -> str:
    cleaned = [truncate_text(value, value_limit) for value in unique_preserve_order(values)]
    return ", ".join(cleaned[:item_limit])


def build_model_enriched_metadata_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Build the richest feasible model representation from available metadata.

    Includes title/label, year, tasks, datasets, keywords, related papers,
    implementations, metrics, runs, linked entities, and description (when present).
    Empty sections are omitted so the text stays dense.
    """
    max_characters = int(config.get("max_characters", 2400))
    title_max = int(config.get("title_max_characters", 512))
    description_max = int(config.get("description_max_characters", 600))
    list_item_limit = int(config.get("list_item_limit", 8))
    list_value_max = int(config.get("list_value_max_characters", 120))
    related_paper_limit = int(config.get("related_paper_limit", 6))
    implementation_limit = int(config.get("implementation_limit", 4))
    dataset_limit = int(config.get("dataset_limit", 6))
    linked_entity_limit = int(config.get("linked_entity_limit", 6))

    parts: List[str] = []

    # --- title / label ---
    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if label:
        parts.append(f"Model: {truncate_text(label, title_max)}")

    # --- year ---
    issued_year = normalize_whitespace(record.get("issued_year") or "")
    if issued_year:
        parts.append(f"Year: {issued_year}")

    # --- tasks ---
    tasks: List[str] = record.get("tasks") or []
    if tasks:
        parts.append(f"Tasks: {_render_list(tasks, list_item_limit, list_value_max)}")

    # --- datasets ---
    datasets: List[str] = record.get("datasets") or []
    if datasets:
        parts.append(f"Datasets: {_render_list(datasets, dataset_limit, list_value_max)}")

    # --- keywords ---
    keywords: List[str] = record.get("keywords") or []
    if keywords:
        parts.append(f"Keywords: {_render_list(keywords, list_item_limit, list_value_max)}")

    # --- related papers ---
    related_papers: List[str] = record.get("related_papers") or []
    if related_papers:
        parts.append(f"Related Papers: {_render_list(related_papers, related_paper_limit, list_value_max)}")

    # --- implementations ---
    related_impls: List[str] = record.get("related_implementations") or []
    if related_impls:
        parts.append(f"Implementations: {_render_list(related_impls, implementation_limit, list_value_max)}")

    # --- metrics ---
    metrics_list: List[str] = record.get("metrics") or []
    if metrics_list:
        parts.append(f"Metrics: {_render_list(metrics_list, list_item_limit, list_value_max)}")

    # --- runs ---
    runs: List[str] = record.get("runs") or []
    if runs:
        parts.append(f"Runs: {_render_list(runs, list_item_limit, list_value_max)}")

    # --- linked entities (structured relations from RDF) ---
    linked_entities: List[Dict[str, Any]] = record.get("linked_entities") or []
    if linked_entities:
        entity_labels = [
            normalize_whitespace(entity.get("object_label") or "")
            for entity in linked_entities
            if normalize_whitespace(entity.get("object_label") or "")
        ]
        if entity_labels:
            parts.append(f"Linked Entities: {_render_list(entity_labels, linked_entity_limit, list_value_max)}")

    # --- description (often empty, added last) ---
    description = normalize_whitespace(record.get("description") or "")
    if description:
        parts.append(f"Description: {truncate_text(description, description_max)}")

    text = "\n".join(parts)
    return truncate_text(text, max_characters)
