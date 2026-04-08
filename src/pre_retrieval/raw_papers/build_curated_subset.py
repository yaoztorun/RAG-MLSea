from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.pre_retrieval.utils import load_json, load_jsonl, normalize_identifier, save_json, save_jsonl


def _load_gold_target_ids(questions_path: Path, include_gold_targets: bool) -> List[str]:
    if not include_gold_targets:
        return []

    questions = load_json(questions_path)
    gold_target_ids: List[str] = []
    seen = set()
    for question in questions:
        if question.get("is_answerable", True) is not True:
            continue
        target_entity_iri = normalize_identifier(str(question.get("target_entity_iri", "")))
        if target_entity_iri and target_entity_iri not in seen:
            seen.add(target_entity_iri)
            gold_target_ids.append(target_entity_iri)
    return gold_target_ids


def build_curated_subset(
    papers_master_path: Path,
    questions_path: Path,
    output_path: Path,
    stats_output_path: Path,
    *,
    max_papers: int,
    include_gold_targets: bool = True,
) -> Dict[str, Any]:
    if max_papers <= 0:
        raise ValueError("max_papers must be greater than 0.")

    master_records = load_jsonl(papers_master_path)
    gold_target_ids = _load_gold_target_ids(questions_path, include_gold_targets=include_gold_targets)
    gold_target_id_set = set(gold_target_ids)

    gold_records: List[Dict[str, Any]] = []
    additional_records: List[Dict[str, Any]] = []
    seen_subset_ids = set()

    for record in master_records:
        paper_id = normalize_identifier(str(record.get("paper_id", "")))
        if paper_id in seen_subset_ids:
            continue
        if paper_id in gold_target_id_set:
            gold_records.append(record)
            seen_subset_ids.add(paper_id)

    missing_gold_targets = [paper_id for paper_id in gold_target_ids if paper_id not in seen_subset_ids]

    subset_records = list(gold_records)
    if len(subset_records) < max_papers:
        remaining_capacity = max_papers - len(subset_records)
        for record in master_records:
            paper_id = normalize_identifier(str(record.get("paper_id", "")))
            if paper_id in seen_subset_ids:
                continue
            additional_records.append(record)
            seen_subset_ids.add(paper_id)
            if len(additional_records) >= remaining_capacity:
                break
        subset_records.extend(additional_records)

    save_jsonl(subset_records, output_path)
    stats = {
        "papers_master_path": str(papers_master_path),
        "questions_path": str(questions_path),
        "subset_output_path": str(output_path),
        "total_master_papers": len(master_records),
        "total_gold_target_papers": len(gold_target_ids),
        "total_subset_papers": len(subset_records),
        "gold_target_papers_included": len(gold_records),
        "additional_fill_papers": max(len(subset_records) - len(gold_records), 0),
        "missing_gold_target_papers": missing_gold_targets,
        "config": {
            "max_papers": max_papers,
            "include_gold_targets": include_gold_targets,
        },
    }
    save_json(stats, stats_output_path)
    return stats
