from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pre_retrieval.raw_papers.build_paper_records import (
    DCTERMS_ISSUED,
    DCTERMS_TITLE,
    DCAT_KEYWORD,
    FOAF_NAME,
    LABEL_PREDICATES,
    MLSO_HAS_TASK_TYPE,
    RDF_TYPE,
    RDFS_LABEL,
    SCHEMA_DATE_PUBLISHED,
    SCHEMA_DESCRIPTION,
    SCHEMA_NAME,
    YEAR_PREDICATES,
    collect_node_metadata_pass2,
    first_value_for_predicates,
    is_debug_input,
    local_name,
    log_progress,
    progress_interval_for_path,
    resolve_node_text,
    resolve_node_types,
    stream_nt_triples,
)
from src.pre_retrieval.utils import (
    normalize_identifier,
    require_existing_input,
    save_json,
    save_jsonl,
    truncate_text,
    unique_preserve_order,
)


DATASET_PREFIX = "http://w3id.org/mlsea/pwc/dataset/"
DCAT_DATASET_TYPE = "http://www.w3.org/ns/dcat#Dataset"
MLS_DATASET_TYPE = "http://w3id.org/mls#Dataset"
MLSO_DATASET_TYPE = "http://w3id.org/mlso/Dataset"

DESCRIPTION_PREDICATES = (SCHEMA_DESCRIPTION,)
KEYWORD_PREDICATES = (DCAT_KEYWORD,)

MLSO_HAS_RELATED_PAPER = "http://w3id.org/mlso/hasRelatedPaper"
MLSO_HAS_RELATED_IMPLEMENTATION = "http://w3id.org/mlso/hasRelatedImplementation"
MLSO_USES_DATASET = "http://w3id.org/mlso/usesDataset"


def is_dataset_subject(subject_uri: str, predicate_uri: str, object_value: str, is_literal: bool) -> bool:
    if subject_uri.startswith(DATASET_PREFIX):
        return True
    if predicate_uri == RDF_TYPE and not is_literal:
        if object_value in {DCAT_DATASET_TYPE, MLS_DATASET_TYPE, MLSO_DATASET_TYPE}:
            return True
    return False


def make_dataset_accumulator(dataset_uri: str) -> Dict[str, Any]:
    return {
        "dataset_uri": dataset_uri,
        "triples": [],
        "raw_predicates": set(),
        "referenced_nodes": set(),
    }


def collect_dataset_triples_pass1(nt_path: Path) -> tuple[Dict[str, Dict[str, Any]], int]:
    progress_interval = progress_interval_for_path(nt_path)
    dataset_map: Dict[str, Dict[str, Any]] = {}
    triples_processed = 0

    for triple in stream_nt_triples(nt_path):
        triples_processed += 1
        subject = triple["subject"]
        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]

        if triples_processed % progress_interval == 0:
            log_progress("build_dataset_records:pass1", triples_processed, len(dataset_map), tracked_label="tracked datasets")

        if not is_dataset_subject(subject, predicate, object_value, is_literal):
            continue

        accumulator = dataset_map.setdefault(subject, make_dataset_accumulator(subject))
        accumulator["triples"].append(
            {
                "predicate": predicate,
                "object": object_value,
                "is_literal": is_literal,
            }
        )
        accumulator["raw_predicates"].add(predicate)
        if not is_literal and not str(object_value).startswith("_:"):
            accumulator["referenced_nodes"].add(object_value)

    log_progress("build_dataset_records:pass1", triples_processed, len(dataset_map), "completed", tracked_label="tracked datasets")
    return dataset_map, triples_processed


def finalize_dataset_record(accumulator: Dict[str, Any], node_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    dataset_uri = accumulator["dataset_uri"]
    triples = accumulator["triples"]

    label = first_value_for_predicates(triples, LABEL_PREDICATES, node_cache)
    title = first_value_for_predicates(triples, (DCTERMS_TITLE,), node_cache) or label
    description = first_value_for_predicates(triples, DESCRIPTION_PREDICATES, node_cache)
    issued_year = first_value_for_predicates(triples, YEAR_PREDICATES, node_cache)

    keywords: List[str] = []
    tasks: List[str] = []
    related_papers: List[str] = []
    related_implementations: List[str] = []
    linked_entities: List[Dict[str, Any]] = []

    core_predicates = set(LABEL_PREDICATES + DESCRIPTION_PREDICATES + YEAR_PREDICATES + KEYWORD_PREDICATES)

    for triple in triples:
        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]

        if predicate in KEYWORD_PREDICATES:
            keywords.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_TASK_TYPE:
            tasks.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_RELATED_PAPER:
            related_papers.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_RELATED_IMPLEMENTATION:
            related_implementations.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate in core_predicates or predicate == RDF_TYPE:
            continue

        resolved_object = resolve_node_text(object_value, is_literal, node_cache)
        if not resolved_object:
            continue

        if not is_literal and not str(object_value).startswith("_:"):
            object_types = resolve_node_types(str(object_value), node_cache)
            linked_entities.append(
                {
                    "predicate": predicate,
                    "predicate_label": local_name(predicate),
                    "object_uri": str(object_value),
                    "object_label": truncate_text(resolved_object, 180),
                    "object_types": object_types,
                }
            )

    return {
        "dataset_id": normalize_identifier(dataset_uri),
        "dataset_uri": dataset_uri,
        "label": label,
        "title": title,
        "description": description,
        "issued_year": issued_year,
        "keywords": unique_preserve_order(keywords),
        "tasks": unique_preserve_order(tasks),
        "related_papers": unique_preserve_order(related_papers),
        "related_implementations": unique_preserve_order(related_implementations),
        "linked_entities": linked_entities,
        "raw_predicates": sorted(accumulator["raw_predicates"]),
    }


def compute_dataset_extraction_stats(records: Sequence[Dict[str, Any]], nt_path: Path, total_triples: int) -> Dict[str, Any]:
    label_lengths = [len(record["label"]) for record in records if record.get("label")]
    description_lengths = [len(record["description"]) for record in records if record.get("description")]
    return {
        "input_path": str(nt_path),
        "total_triples": total_triples,
        "total_datasets": len(records),
        "datasets_with_label": sum(1 for record in records if record.get("label")),
        "datasets_with_title": sum(1 for record in records if record.get("title")),
        "datasets_with_description": sum(1 for record in records if record.get("description")),
        "datasets_with_tasks": sum(1 for record in records if record.get("tasks")),
        "datasets_with_keywords": sum(1 for record in records if record.get("keywords")),
        "avg_label_length": (sum(label_lengths) / len(label_lengths)) if label_lengths else 0.0,
        "avg_description_length": (sum(description_lengths) / len(description_lengths)) if description_lengths else 0.0,
    }


def build_dataset_records(
    nt_path: Path,
    output_path: Path,
    extraction_stats_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    require_existing_input(nt_path)
    if is_debug_input(nt_path):
        print("[build_dataset_records] sample/debug input detected; streaming with frequent progress updates.", flush=True)

    dataset_map, total_triples = collect_dataset_triples_pass1(nt_path)
    selected_dataset_uris = sorted(dataset_map)
    if limit is not None:
        selected_dataset_uris = selected_dataset_uris[:limit]

    referenced_nodes = sorted(
        {
            node_uri
            for dataset_uri in selected_dataset_uris
            for node_uri in dataset_map[dataset_uri]["referenced_nodes"]
        }
    )
    node_cache = collect_node_metadata_pass2(nt_path, referenced_nodes)

    records = [finalize_dataset_record(dataset_map[dataset_uri], node_cache) for dataset_uri in selected_dataset_uris]
    save_jsonl(records, output_path)

    extraction_stats = compute_dataset_extraction_stats(records, nt_path, total_triples)
    save_json(extraction_stats, extraction_stats_path)
    return extraction_stats
