from __future__ import annotations

from typing import Any, Dict, List

from src.pre_retrieval.shared.utils import truncate_text


def build_title_abstract_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = truncate_text(record.get("title"), int(config.get("title_max_characters", 512)))
    abstract = truncate_text(record.get("abstract"), int(config.get("abstract_max_characters", 1400)))
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return truncate_text("\n".join(parts), int(config.get("max_characters", 1800)))
