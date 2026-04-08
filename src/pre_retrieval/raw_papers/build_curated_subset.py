from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.pre_retrieval.utils import (
    entity_type_from_id,
    is_paper_entity_id,
    load_json,
    load_jsonl,
    normalize_identifier,
    save_json,
    save_jsonl,
)


def _load_gold_targets(questions_path: Path) -> List[Dict[str, str]]:
    questions = load_json(questions_path)
    gold_targets: List[Dict[str, str]] = []
    seen = set()
    for question in questions:
        if question.get("is_answerable", True) is not True:
            continue
        target_entity_iri = normalize_identifier(str(question.get("target_entity_iri", "")))
        if target_entity_iri and target_entity_iri not in seen:
            seen.add(target_entity_iri)
            gold_targets.append(
                {
                    "entity_id": target_entity_iri,
                    "entity_type": entity_type_from_id(target_entity_iri),
                }
            )
    return gold_targets


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
    gold_targets = _load_gold_targets(questions_path)
    paper_gold_target_ids = [
        target["entity_id"]
        for target in gold_targets
        if is_paper_entity_id(target["entity_id"])
    ]
    required_paper_gold_target_ids = paper_gold_target_ids if include_gold_targets else []
    skipped_non_paper_targets = [
        target["entity_id"]
        for target in gold_targets
        if not is_paper_entity_id(target["entity_id"])
    ]
    required_paper_gold_target_id_set = set(required_paper_gold_target_ids)

    gold_records: List[Dict[str, Any]] = []
    additional_records: List[Dict[str, Any]] = []
    seen_subset_ids = set()

    for record in master_records:
        paper_id = normalize_identifier(str(record.get("paper_id", "")))
        if paper_id in seen_subset_ids:
            continue
        if paper_id in required_paper_gold_target_id_set:
            gold_records.append(record)
            seen_subset_ids.add(paper_id)

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

    subset_paper_ids = {normalize_identifier(str(record.get("paper_id", ""))) for record in subset_records}
    gold_target_papers_included = sum(1 for paper_id in paper_gold_target_ids if paper_id in subset_paper_ids)
    missing_gold_target_papers = [paper_id for paper_id in paper_gold_target_ids if paper_id not in subset_paper_ids]

    save_jsonl(subset_records, output_path)
    stats = {
        "papers_master_path": str(papers_master_path),
        "questions_path": str(questions_path),
        "subset_output_path": str(output_path),
        "total_master_papers": len(master_records),
        "total_gold_targets": len(gold_targets),
        "gold_target_papers": len(paper_gold_target_ids),
        "gold_target_non_papers": len(skipped_non_paper_targets),
        "total_subset_papers": len(subset_records),
        "gold_target_papers_included": gold_target_papers_included,
        "additional_fill_papers": max(len(subset_records) - len(gold_records), 0),
        "missing_gold_target_papers": missing_gold_target_papers,
        "skipped_non_paper_targets": skipped_non_paper_targets,
        "config": {
            "max_papers": max_papers,
            "include_gold_targets": include_gold_targets,
        },
    }
    save_json(stats, stats_output_path)
    return stats
