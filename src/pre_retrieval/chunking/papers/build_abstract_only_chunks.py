from __future__ import annotations

from typing import Any, Dict

from src.pre_retrieval.utils import truncate_text


def build_abstract_only_text(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    return truncate_text(record.get("abstract"), int(config.get("max_characters", 1600)))
