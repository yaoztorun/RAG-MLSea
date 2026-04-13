from __future__ import annotations

from typing import Any, Dict, List

from src.pre_retrieval.models.chunking.model_graph_helpers import (
    extract_neighbor_labels,
    extract_repo_names,
    extract_repo_urls,
)
from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text, unique_preserve_order


def build_model_predicate_filtered_text(record: Dict[str, Any], config: Dict[str, Any]) -> str | None:
    """Build predicate-filtered model representation.

    Drops models that have NO linked_entities **and** no non-empty top-level
    relation lists (tasks, datasets, metrics, etc.).  This keeps models that
    actually carry graph-derived context.
    """
    max_characters = int(config.get("max_characters", 1800))
    title_max = int(config.get("title_max_characters", 512))
    list_item_limit = int(config.get("list_item_limit", 5))
    list_value_max = int(config.get("list_value_max_characters", 100))

    linked_entities: List[Dict[str, Any]] = record.get("linked_entities") or []

    # Check if there is *any* useful graph context beyond a bare label.
    top_level_lists = [
        record.get(f) or []
        for f in ("tasks", "datasets", "metrics", "runs", "keywords",
                   "related_papers", "related_implementations")
    ]
    has_graph_context = bool(linked_entities) or any(top_level_lists)
    if not has_graph_context:
        return None  # DROP model – nothing useful beyond title

    parts: List[str] = []

    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if label:
        parts.append(f"Model: {truncate_text(label, title_max)}")

    # --- repository names ---
    repo_names = extract_repo_names(linked_entities)
    if repo_names:
        items = [truncate_text(name, list_value_max) for name in repo_names[:list_item_limit]]
        parts.append(f"Repositories: {', '.join(items)}")

    # --- repository URLs ---
    repo_urls = extract_repo_urls(linked_entities)
    if repo_urls:
        items = [truncate_text(url, list_value_max) for url in repo_urls[:list_item_limit]]
        parts.append(f"Repository URLs: {', '.join(items)}")

    # --- graph-linked neighbor labels ---
    neighbor_labels = extract_neighbor_labels(linked_entities)
    if neighbor_labels:
        items = [truncate_text(lbl, list_value_max) for lbl in neighbor_labels[:list_item_limit]]
        parts.append(f"Linked Entities: {', '.join(items)}")

    # --- include any populated top-level lists ---
    for field, heading in [
        ("tasks", "Tasks"),
        ("datasets", "Datasets"),
        ("related_papers", "Related Papers"),
        ("related_implementations", "Implementations"),
        ("metrics", "Metrics"),
    ]:
        values: List[str] = record.get(field) or []
        if values:
            items = [truncate_text(v, list_value_max) for v in unique_preserve_order(values)[:list_item_limit]]
            parts.append(f"{heading}: {', '.join(items)}")

    text = "\n".join(parts)
    return truncate_text(text, max_characters)
