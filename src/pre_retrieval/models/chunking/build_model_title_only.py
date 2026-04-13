from __future__ import annotations

from typing import Any, Dict

from src.pre_retrieval.shared.utils import normalize_whitespace, truncate_text


def build_model_title_only_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Minimal model representation: ``Model: <label>``."""
    max_characters = int(config.get("max_characters", 512))
    label = normalize_whitespace(record.get("label") or record.get("title") or "")
    if not label:
        return ""
    return truncate_text(f"Model: {label}", max_characters)
