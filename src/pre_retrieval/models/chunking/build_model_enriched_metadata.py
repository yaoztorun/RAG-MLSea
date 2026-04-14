from __future__ import annotations

import re
from typing import Any, Dict, List

from src.pre_retrieval.models.chunking.model_graph_helpers import (
    extract_neighbor_labels,
    extract_repo_names,
    extract_repo_urls,
)
from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text, unique_preserve_order


_FAMILY_PATTERN = re.compile(
    r"^(.*?)\s*[-_]?\s*(?:Small|Base|Medium|Large|XL|XXL|Tiny|Nano|Huge|Mini|v\d+(?:\.\d+)*|[0-9]+[BbMmKk]?)\s*$"
)


def _infer_model_family(label: str) -> str:
    """Heuristically extract a model family name from the label.

    Example: ``GPT-2 Small`` → ``GPT-2``, ``ResNet-50`` → ``ResNet``.
    Returns empty string when no family can be extracted or it would be the
    same as the original label.
    """
    if not label:
        return ""
    match = _FAMILY_PATTERN.match(label.strip())
    if match:
        family = match.group(1).strip().rstrip("-_ ")
        if family and family != label.strip():
            return family
    return ""


def _render_list(values: list[str], item_limit: int, value_limit: int) -> str:
    cleaned = [truncate_text(value, value_limit) for value in unique_preserve_order(values)]
    return ", ".join(cleaned[:item_limit])


def build_model_enriched_metadata_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Build the richest feasible model representation.

    Because models in the KG are NOT text-rich, this representation draws
    primarily from:
    - label / title
    - model family heuristic
    - linked_entities → repository names, URLs, neighbor labels
    - any populated top-level lists (rare but included when present)
    - raw_predicates summary (as lightweight structural signal)
    """
    max_characters = int(config.get("max_characters", 2400))
    title_max = int(config.get("title_max_characters", 512))
    description_max = int(config.get("description_max_characters", 600))
    list_item_limit = int(config.get("list_item_limit", 8))
    list_value_max = int(config.get("list_value_max_characters", 120))
    linked_entity_limit = int(config.get("linked_entity_limit", 6))

    parts: List[str] = []

    # --- title / label ---
    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if not label:
        return ""
    parts.append(f"Model: {truncate_text(label, title_max)}")

    # --- model family heuristic ---
    family = _infer_model_family(label)
    if family:
        parts.append(f"Model Family: {family}")

    linked_entities: List[Dict[str, Any]] = record.get("linked_entities") or []

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
        items = [truncate_text(lbl, list_value_max) for lbl in neighbor_labels[:linked_entity_limit]]
        parts.append(f"Linked Entities: {', '.join(items)}")

    # --- top-level lists (often empty, but include when populated) ---
    for field, heading in [
        ("tasks", "Tasks"),
        ("datasets", "Datasets"),
        ("keywords", "Keywords"),
        ("related_papers", "Related Papers"),
        ("related_implementations", "Implementations"),
        ("metrics", "Metrics"),
        ("runs", "Runs"),
    ]:
        values: List[str] = record.get(field) or []
        if values:
            items = [truncate_text(v, list_value_max) for v in unique_preserve_order(values)[:list_item_limit]]
            parts.append(f"{heading}: {', '.join(items)}")

    # --- description (often empty) ---
    description = normalize_whitespace(record.get("description") or "")
    if description:
        parts.append(f"Description: {truncate_text(description, description_max)}")

    # --- raw predicates summary (lightweight structural signal) ---
    raw_predicates: List[str] = record.get("raw_predicates") or []
    if raw_predicates:
        predicate_names = unique_preserve_order(
            p.rsplit("/", 1)[-1].rsplit("#", 1)[-1] for p in raw_predicates if p
        )
        if predicate_names:
            items = [truncate_text(p, list_value_max) for p in predicate_names[:list_item_limit]]
            parts.append(f"Predicates: {', '.join(items)}")

    text = "\n".join(parts)
    return truncate_text(text, max_characters)