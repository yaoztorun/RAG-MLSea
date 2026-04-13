from __future__ import annotations

from typing import Any, Dict

from src.pre_retrieval.utils import normalize_whitespace, truncate_text


def build_dataset_title_only_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    max_characters = int(config.get("max_characters", 512))
    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    return truncate_text(label, max_characters)
