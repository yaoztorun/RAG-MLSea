from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pre_retrieval.papers.raw.build_paper_records import (
    DCTERMS_ISSUED,
    DCTERMS_TITLE,
    DCAT_KEYWORD,
    LABEL_PREDICATES,
    MLSO_HAS_TASK_TYPE,
    RDF_TYPE,
    RDFS_LABEL,
    SCHEMA_DATE_PUBLISHED,
    SCHEMA_DESCRIPTION,
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
from src.pre_retrieval.shared.utils import (
    normalize_identifier,
    require_existing_input,
    save_json,
    save_jsonl,
    truncate_text,
    unique_preserve_order,
)


MODEL_PREFIX = "http://w3id.org/mlsea/pwc/model/"
MLS_MODEL_TYPE = "http://w3id.org/mls#Model"
MLSO_MODEL_TYPE = "http://w3id.org/mlso/Model"

DESCRIPTION_PREDICATES = (SCHEMA_DESCRIPTION,)
KEYWORD_PREDICATES = (DCAT_KEYWORD,)

MLSO_HAS_RELATED_PAPER = "http://w3id.org/mlso/hasRelatedPaper"
MLSO_HAS_RELATED_IMPLEMENTATION = "http://w3id.org/mlso/hasRelatedImplementation"
MLSO_USES_DATASET = "http://w3id.org/mlso/usesDataset"
MLS_HAS_MODEL_CHARACTERISTIC = "http://w3id.org/mls#hasModelCharacteristic"
MLS_REALIZES = "http://w3id.org/mls#realizes"
MLS_HAS_HYPERPARAMETER = "http://w3id.org/mls#hasHyperParameter"
MLSO_HAS_EVALUATION = "http://w3id.org/mlso/hasEvaluation"
MLSO_HAS_RUN = "http://w3id.org/mlso/hasRun"
MLS_HAS_INPUT = "http://w3id.org/mls#hasInput"
MLS_HAS_OUTPUT = "http://w3id.org/mls#hasOutput"


def is_model_subject(subject_uri: str, predicate_uri: str, object_value: str, is_literal: bool) -> bool:
    if subject_uri.startswith(MODEL_PREFIX):
        return True
    if predicate_uri == RDF_TYPE and not is_literal:
        if object_value in {MLS_MODEL_TYPE, MLSO_MODEL_TYPE}:
            return True
    return False


def make_model_accumulator(model_uri: str) -> Dict[str, Any]:
    return {
        "model_uri": model_uri,
        "triples": [],
        "raw_predicates": set(),
        "referenced_nodes": set(),
    }


def collect_model_triples_pass1(nt_path: Path) -> tuple[Dict[str, Dict[str, Any]], int]:
    progress_interval = progress_interval_for_path(nt_path)
    model_map: Dict[str, Dict[str, Any]] = {}
    triples_processed = 0

    for triple in stream_nt_triples(nt_path):
        triples_processed += 1
        subject = triple["subject"]
        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]

        if triples_processed % progress_interval == 0:
            log_progress("build_model_records:pass1", triples_processed, len(model_map), tracked_label="tracked models")

        if not is_model_subject(subject, predicate, object_value, is_literal):
            continue

        accumulator = model_map.setdefault(subject, make_model_accumulator(subject))
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

    log_progress("build_model_records:pass1", triples_processed, len(model_map), "completed", tracked_label="tracked models")
    return model_map, triples_processed


def finalize_model_record(accumulator: Dict[str, Any], node_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    model_uri = accumulator["model_uri"]
    triples = accumulator["triples"]

    label = first_value_for_predicates(triples, LABEL_PREDICATES, node_cache)
    title = first_value_for_predicates(triples, (DCTERMS_TITLE,), node_cache) or label
    description = first_value_for_predicates(triples, DESCRIPTION_PREDICATES, node_cache)
    issued_year = first_value_for_predicates(triples, YEAR_PREDICATES, node_cache)

    keywords: List[str] = []
    tasks: List[str] = []
    datasets: List[str] = []
    related_papers: List[str] = []
    related_implementations: List[str] = []
    runs: List[str] = []
    metrics: List[str] = []
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

        if predicate == MLSO_USES_DATASET:
            datasets.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_RELATED_PAPER:
            related_papers.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_RELATED_IMPLEMENTATION:
            related_implementations.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_RUN:
            runs.append(resolve_node_text(object_value, is_literal, node_cache))
            continue

        if predicate == MLSO_HAS_EVALUATION:
            metrics.append(resolve_node_text(object_value, is_literal, node_cache))
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
        "model_id": normalize_identifier(model_uri),
        "model_uri": model_uri,
        "label": label,
        "title": title,
        "description": description,
        "issued_year": issued_year,
        "keywords": unique_preserve_order(keywords),
        "tasks": unique_preserve_order(tasks),
        "datasets": unique_preserve_order(datasets),
        "related_papers": unique_preserve_order(related_papers),
        "related_implementations": unique_preserve_order(related_implementations),
        "runs": unique_preserve_order(runs),
        "metrics": unique_preserve_order(metrics),
        "linked_entities": linked_entities,
        "raw_predicates": sorted(accumulator["raw_predicates"]),
    }


def _merge_accumulators(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Merge *source* accumulator into *target* in-place."""
    target["triples"].extend(source["triples"])
    target["raw_predicates"].update(source["raw_predicates"])
    target["referenced_nodes"].update(source["referenced_nodes"])


def deduplicate_model_map(model_map: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], int]:
    """Merge accumulators whose URIs normalise to the same ``model_id``.

    Returns the deduplicated map (keyed by the first raw URI encountered per
    normalised ID) and the number of duplicate raw URIs that were merged.
    """
    seen: Dict[str, str] = {}  # normalized_id -> canonical raw URI
    merged_count = 0
    for raw_uri in list(model_map):
        normalized = normalize_identifier(raw_uri)
        if normalized in seen:
            canonical_uri = seen[normalized]
            _merge_accumulators(model_map[canonical_uri], model_map.pop(raw_uri))
            merged_count += 1
        else:
            seen[normalized] = raw_uri
    return model_map, merged_count


def compute_model_extraction_stats(
    records: Sequence[Dict[str, Any]],
    nt_path: Path,
    total_triples: int,
    *,
    duplicate_model_ids_merged: int = 0,
) -> Dict[str, Any]:
    label_lengths = [len(record["label"]) for record in records if record.get("label")]
    description_lengths = [len(record["description"]) for record in records if record.get("description")]
    return {
        "input_path": str(nt_path),
        "total_triples": total_triples,
        "total_models": len(records),
        "duplicate_model_ids_detected": duplicate_model_ids_merged,
        "duplicate_model_ids_merged": duplicate_model_ids_merged,
        "models_with_label": sum(1 for record in records if record.get("label")),
        "models_with_title": sum(1 for record in records if record.get("title")),
        "models_with_description": sum(1 for record in records if record.get("description")),
        "models_with_tasks": sum(1 for record in records if record.get("tasks")),
        "models_with_datasets": sum(1 for record in records if record.get("datasets")),
        "models_with_keywords": sum(1 for record in records if record.get("keywords")),
        "models_with_metrics": sum(1 for record in records if record.get("metrics")),
        "models_with_implementations": sum(1 for record in records if record.get("related_implementations")),
        "models_with_related_papers": sum(1 for record in records if record.get("related_papers")),
        "models_with_runs": sum(1 for record in records if record.get("runs")),
        "avg_label_length": (sum(label_lengths) / len(label_lengths)) if label_lengths else 0.0,
        "avg_description_length": (sum(description_lengths) / len(description_lengths)) if description_lengths else 0.0,
    }


def build_model_records(
    nt_path: Path,
    output_path: Path,
    extraction_stats_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    require_existing_input(nt_path)
    if is_debug_input(nt_path):
        print("[build_model_records] sample/debug input detected; streaming with frequent progress updates.", flush=True)

    model_map, total_triples = collect_model_triples_pass1(nt_path)

    # Merge accumulators whose raw URIs normalise to the same model_id
    model_map, duplicate_model_ids_merged = deduplicate_model_map(model_map)
    if duplicate_model_ids_merged:
        print(f"[build_model_records] merged {duplicate_model_ids_merged} duplicate model URI(s) by normalized ID.", flush=True)

    selected_model_uris = sorted(model_map)
    if limit is not None:
        selected_model_uris = selected_model_uris[:limit]

    referenced_nodes = sorted(
        {
            node_uri
            for model_uri in selected_model_uris
            for node_uri in model_map[model_uri]["referenced_nodes"]
        }
    )
    node_cache = collect_node_metadata_pass2(nt_path, referenced_nodes)

    records = [finalize_model_record(model_map[model_uri], node_cache) for model_uri in selected_model_uris]
    save_jsonl(records, output_path)

    extraction_stats = compute_model_extraction_stats(
        records, nt_path, total_triples,
        duplicate_model_ids_merged=duplicate_model_ids_merged,
    )
    save_json(extraction_stats, extraction_stats_path)
    return extraction_stats
