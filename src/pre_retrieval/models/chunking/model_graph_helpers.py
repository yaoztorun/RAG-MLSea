"""Shared helpers for extracting useful graph context from model records.

Model records in ``models_master.jsonl`` are sparse – most top-level fields
(description, tasks, datasets, …) are empty.  The dominant useful information
lives in ``linked_entities``, especially ``codeRepository`` links.

These helpers normalise repository URLs into human-readable names and extract
the set of unique neighbour labels from ``linked_entities``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.pre_retrieval.shared.utils import normalize_whitespace, unique_preserve_order

# URI / predicate fragments that signal a code-repository link.
_REPO_PREDICATES = {"codeRepository", "code_repository", "coderepository"}
_REPO_HOST_PATTERN = re.compile(r"^https?://(?:www\.)?(github|gitlab|bitbucket)\.")


def _is_repo_entity(entity: Dict[str, Any]) -> bool:
    """Return True if the linked entity looks like a code-repository link."""
    predicate_label = normalize_whitespace(entity.get("predicate_label") or "").lower()
    if predicate_label in _REPO_PREDICATES:
        return True
    object_label = normalize_whitespace(entity.get("object_label") or "")
    if _REPO_HOST_PATTERN.match(object_label):
        return True
    object_uri = normalize_whitespace(entity.get("object_uri") or "")
    if _REPO_HOST_PATTERN.match(object_uri):
        return True
    return False


def _repo_url(entity: Dict[str, Any]) -> str:
    """Return the best URL string for a repo entity."""
    for key in ("object_uri", "object_label"):
        val = normalize_whitespace(entity.get(key) or "")
        if val.startswith("http"):
            return val
    return ""


def _repo_name_from_url(url: str) -> str:
    """Extract a readable ``owner/repo`` slug from a GitHub / GitLab URL.

    Falls back to the last path segment(s) when the host isn't recognised.
    """
    url = url.rstrip("/")
    parts = url.split("/")
    # https://github.com/owner/repo → owner/repo
    if len(parts) >= 5 and _REPO_HOST_PATTERN.match(url):
        return "/".join(parts[3:5])
    if len(parts) >= 4:
        return parts[-1]
    return url


def extract_repo_urls(linked_entities: List[Dict[str, Any]]) -> List[str]:
    """Return deduplicated repository URLs from linked entities."""
    urls: List[str] = []
    for entity in linked_entities:
        if _is_repo_entity(entity):
            url = _repo_url(entity)
            if url:
                urls.append(url)
    return unique_preserve_order(urls)


def extract_repo_names(linked_entities: List[Dict[str, Any]]) -> List[str]:
    """Return deduplicated human-readable repository names."""
    names: List[str] = []
    for url in extract_repo_urls(linked_entities):
        name = _repo_name_from_url(url)
        if name:
            names.append(name)
    return unique_preserve_order(names)


def extract_neighbor_labels(linked_entities: List[Dict[str, Any]]) -> List[str]:
    """Return deduplicated non-URL neighbour labels from linked entities.

    Skips entities that are purely repository links (already surfaced
    separately) and entities whose label looks like a raw URI.
    """
    labels: List[str] = []
    for entity in linked_entities:
        if _is_repo_entity(entity):
            continue
        label = normalize_whitespace(entity.get("object_label") or "")
        if not label or label.startswith("http://") or label.startswith("https://"):
            continue
        labels.append(label)
    return unique_preserve_order(labels)
