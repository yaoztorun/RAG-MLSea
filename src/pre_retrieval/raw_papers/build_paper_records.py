from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from src.pre_retrieval.utils import paper_id_from_uri, require_existing_input, save_json, save_jsonl, truncate_text, unique_preserve_order


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
FOAF_NAME = "http://xmlns.com/foaf/0.1/name"
DCTERMS_TITLE = "http://purl.org/dc/terms/title"
DCTERMS_CREATOR = "http://purl.org/dc/terms/creator"
DCTERMS_ISSUED = "http://purl.org/dc/terms/issued"
FABIO_ABSTRACT = "http://purl.org/spar/fabio/abstract"
DCAT_KEYWORD = "http://www.w3.org/ns/dcat#keyword"
SCHEMA_NAME = "https://schema.org/name"
SCHEMA_DESCRIPTION = "https://schema.org/description"
SCHEMA_DATE_PUBLISHED = "https://schema.org/datePublished"
SCHEMA_AUTHOR = "https://schema.org/author"
SCHEMA_CODE_REPOSITORY = "https://schema.org/codeRepository"

MLSO_SCIENTIFIC_WORK = "http://w3id.org/mlso/ScientificWork"
MLSO_HAS_TASK_TYPE = "http://w3id.org/mlso/hasTaskType"
MLSO_HAS_RELATED_IMPLEMENTATION = "http://w3id.org/mlso/hasRelatedImplementation"

PAPER_PREFIX = "http://w3id.org/mlsea/pwc/scientificWork/"
LABEL_PREDICATES = (DCTERMS_TITLE, RDFS_LABEL, FOAF_NAME, SCHEMA_NAME)
ABSTRACT_PREDICATES = (FABIO_ABSTRACT, SCHEMA_DESCRIPTION)
YEAR_PREDICATES = (DCTERMS_ISSUED, SCHEMA_DATE_PUBLISHED)
AUTHOR_PREDICATES = (DCTERMS_CREATOR, SCHEMA_AUTHOR)
KEYWORD_PREDICATES = (DCAT_KEYWORD,)
FIELD_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "title": LABEL_PREDICATES,
    "abstract": ABSTRACT_PREDICATES,
    "authors": AUTHOR_PREDICATES,
    "keywords": KEYWORD_PREDICATES,
    "tasks": (MLSO_HAS_TASK_TYPE,),
    "implementations": (MLSO_HAS_RELATED_IMPLEMENTATION, SCHEMA_CODE_REPOSITORY),
}
CATEGORY_TOKEN_MAP = {
    "tasks": {"task"},
    "datasets": {"dataset", "benchmark", "corpus"},
    "methods": {"method", "model", "architecture", "algorithm", "approach"},
    "metrics": {"metric", "measure", "score"},
    "implementations": {"implementation", "repository", "github", "code"},
}
CORE_METADATA_PREDICATES = set(LABEL_PREDICATES + ABSTRACT_PREDICATES + YEAR_PREDICATES + AUTHOR_PREDICATES + KEYWORD_PREDICATES)
NTRIPLE_PATTERN = re.compile(r'^<(?P<subject>[^>]*)>\s+<(?P<predicate>[^>]*)>\s+(?P<object>.+?)\s*\.\s*$')
LITERAL_PATTERN = re.compile(r'^"(?P<value>(?:[^"\\]|\\.)*)"(?:@(?P<language>[A-Za-z0-9\-]+)|\^\^<(?P<datatype>[^>]*)>)?$')
# Use sparse progress updates for multi-GB production dumps and more frequent updates for sample/debug files.
PROGRESS_INTERVAL = 500_000
DEBUG_PROGRESS_INTERVAL = 10_000
# Treat small files as debug/sample inputs so progress stays chatty during development but quieter on large dumps.
DEBUG_FILE_SIZE_BYTES = 25 * 1024 * 1024


def local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri


def _decode_escaped_literal(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def parse_ntriple_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    match = NTRIPLE_PATTERN.match(stripped)
    if not match:
        return None

    subject = match.group("subject")
    predicate = match.group("predicate")
    object_token = match.group("object")
    if object_token.startswith("<") and object_token.endswith(">"):
        return {
            "subject": subject,
            "predicate": predicate,
            "object": object_token[1:-1],
            "is_literal": False,
        }

    literal_match = LITERAL_PATTERN.match(object_token)
    if literal_match:
        return {
            "subject": subject,
            "predicate": predicate,
            "object": _decode_escaped_literal(literal_match.group("value")),
            "is_literal": True,
        }

    if object_token.startswith("_:"):
        return {
            "subject": subject,
            "predicate": predicate,
            "object": object_token,
            "is_literal": False,
        }
    return None


def is_debug_input(nt_path: Path) -> bool:
    file_name = nt_path.name.lower()
    if "sample" in file_name:
        return True
    if not nt_path.exists():
        return False
    return nt_path.stat().st_size <= DEBUG_FILE_SIZE_BYTES


def progress_interval_for_path(nt_path: Path) -> int:
    return DEBUG_PROGRESS_INTERVAL if is_debug_input(nt_path) else PROGRESS_INTERVAL


def log_progress(
    stage: str,
    triples_processed: int,
    tracked_count: int,
    extra: str = "",
    tracked_label: str = "tracked papers",
) -> None:
    suffix = f" {extra}" if extra else ""
    print(f"[{stage}] processed {triples_processed:,} triples; {tracked_label}={tracked_count:,}{suffix}", flush=True)


def is_paper_subject(subject_uri: str, predicate_uri: str, object_value: str, is_literal: bool) -> bool:
    return subject_uri.startswith(PAPER_PREFIX) or (
        predicate_uri == RDF_TYPE and not is_literal and object_value == MLSO_SCIENTIFIC_WORK
    )


def make_paper_accumulator(paper_uri: str) -> Dict[str, Any]:
    return {
        "paper_uri": paper_uri,
        "triples": [],
        "raw_predicates": set(),
        "referenced_nodes": set(),
    }


def stream_nt_triples(nt_path: Path) -> Iterator[Dict[str, Any]]:
    require_existing_input(nt_path)
    with nt_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = parse_ntriple_line(line)
            if parsed is None:
                continue
            parsed["line_number"] = line_number
            yield parsed


def collect_paper_triples_pass1(nt_path: Path) -> tuple[Dict[str, Dict[str, Any]], int]:
    progress_interval = progress_interval_for_path(nt_path)
    paper_map: Dict[str, Dict[str, Any]] = {}
    triples_processed = 0

    for triple in stream_nt_triples(nt_path):
        triples_processed += 1
        subject = triple["subject"]
        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]

        if triples_processed % progress_interval == 0:
            log_progress("build_records:pass1", triples_processed, len(paper_map))

        if not is_paper_subject(subject, predicate, object_value, is_literal):
            continue

        accumulator = paper_map.setdefault(subject, make_paper_accumulator(subject))
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

    log_progress("build_records:pass1", triples_processed, len(paper_map), "completed")
    return paper_map, triples_processed


def collect_node_metadata_pass2(nt_path: Path, referenced_nodes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    progress_interval = progress_interval_for_path(nt_path)
    tracked_nodes = set(referenced_nodes)
    node_cache: Dict[str, Dict[str, Any]] = {}
    triples_processed = 0

    if not tracked_nodes:
        return node_cache

    for triple in stream_nt_triples(nt_path):
        triples_processed += 1
        if triples_processed % progress_interval == 0:
            log_progress(
                "build_records:pass2",
                triples_processed,
                len(tracked_nodes),
                f"cached_nodes={len(node_cache):,}",
                tracked_label="tracked nodes",
            )

        subject = triple["subject"]
        if subject not in tracked_nodes:
            continue

        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]
        node_info = node_cache.setdefault(subject, {"labels": {}, "types": []})

        if predicate in LABEL_PREDICATES and is_literal:
            node_info["labels"].setdefault(predicate, []).append(object_value)
        elif predicate == RDF_TYPE and not is_literal:
            node_info["types"].append(object_value)

    log_progress(
        "build_records:pass2",
        triples_processed,
        len(tracked_nodes),
        f"cached_nodes={len(node_cache):,} completed",
        tracked_label="tracked nodes",
    )
    return node_cache


def resolve_node_text(node_value: str, is_literal: bool, node_cache: Dict[str, Dict[str, Any]]) -> str:
    if is_literal:
        return str(node_value).strip()
    if str(node_value).startswith("_:"):
        return str(node_value).strip()

    node_info = node_cache.get(node_value, {})
    labels = node_info.get("labels", {})
    for predicate in LABEL_PREDICATES:
        predicate_labels = labels.get(predicate, [])
        if predicate_labels:
            return str(predicate_labels[0]).strip()
    return local_name(str(node_value)).replace("_", " ").strip()


def resolve_node_types(node_uri: str, node_cache: Dict[str, Dict[str, Any]]) -> List[str]:
    node_info = node_cache.get(node_uri, {})
    return unique_preserve_order(local_name(node_type).lower() for node_type in node_info.get("types", []))


def first_value_for_predicates(
    triples: Sequence[Dict[str, Any]],
    predicates: Iterable[str],
    node_cache: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    for predicate in predicates:
        for triple in triples:
            if triple["predicate"] != predicate:
                continue
            value = resolve_node_text(triple["object"], triple["is_literal"], node_cache)
            if value:
                return value
    return None


def infer_bucket(predicate: str, object_node: str, object_types: List[str]) -> Optional[str]:
    predicate_tokens = local_name(predicate).lower()
    object_tokens = f"{local_name(object_node).lower()} {' '.join(object_types)}"
    if predicate == MLSO_HAS_TASK_TYPE:
        return "tasks"
    if predicate in {MLSO_HAS_RELATED_IMPLEMENTATION, SCHEMA_CODE_REPOSITORY}:
        return "implementations"
    for bucket, tokens in CATEGORY_TOKEN_MAP.items():
        if any(token in predicate_tokens or token in object_tokens for token in tokens):
            return bucket
    return None


def finalize_paper_record(accumulator: Dict[str, Any], node_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    paper_uri = accumulator["paper_uri"]
    triples = accumulator["triples"]
    title = first_value_for_predicates(triples, LABEL_PREDICATES, node_cache)
    abstract = first_value_for_predicates(triples, ABSTRACT_PREDICATES, node_cache)
    year = first_value_for_predicates(triples, YEAR_PREDICATES, node_cache)

    authors: List[str] = []
    keywords: List[str] = []
    tasks: List[str] = []
    datasets: List[str] = []
    methods: List[str] = []
    metrics: List[str] = []
    implementations: List[str] = []
    linked_entities: List[Dict[str, Any]] = []

    for triple in triples:
        predicate = triple["predicate"]
        object_value = triple["object"]
        is_literal = triple["is_literal"]

        if predicate in AUTHOR_PREDICATES:
            authors.append(resolve_node_text(object_value, is_literal, node_cache))
            continue
        if predicate in KEYWORD_PREDICATES:
            keywords.append(resolve_node_text(object_value, is_literal, node_cache))
            continue
        if predicate in CORE_METADATA_PREDICATES or predicate == RDF_TYPE:
            continue

        resolved_object = resolve_node_text(object_value, is_literal, node_cache)
        if not resolved_object:
            continue

        if not is_literal and not str(object_value).startswith("_:"):
            object_types = resolve_node_types(str(object_value), node_cache)
            category = infer_bucket(predicate, str(object_value), object_types)
            if category == "tasks":
                tasks.append(resolved_object)
            elif category == "datasets":
                datasets.append(resolved_object)
            elif category == "methods":
                methods.append(resolved_object)
            elif category == "metrics":
                metrics.append(resolved_object)
            elif category == "implementations":
                implementations.append(resolved_object)

            linked_entities.append(
                {
                    "predicate": predicate,
                    "predicate_label": local_name(predicate),
                    "object_uri": str(object_value),
                    "object_label": truncate_text(resolved_object, 180),
                    "object_types": object_types,
                    "category": category or "linked_entity",
                }
            )
        elif predicate == SCHEMA_CODE_REPOSITORY:
            implementations.append(resolved_object)

    return {
        "paper_id": paper_id_from_uri(paper_uri),
        "paper_uri": paper_uri,
        "title": title,
        "year": year,
        "abstract": abstract,
        "authors": unique_preserve_order(authors),
        "keywords": unique_preserve_order(keywords),
        "tasks": unique_preserve_order(tasks),
        "datasets": unique_preserve_order(datasets),
        "methods": unique_preserve_order(methods),
        "metrics": unique_preserve_order(metrics),
        "implementations": unique_preserve_order(implementations),
        "linked_entities": linked_entities,
        "raw_predicates": sorted(accumulator["raw_predicates"]),
    }


def compute_extraction_stats(records: Sequence[Dict[str, Any]], nt_path: Path, total_triples: int) -> Dict[str, Any]:
    title_lengths = [len(record["title"]) for record in records if record.get("title")]
    abstract_lengths = [len(record["abstract"]) for record in records if record.get("abstract")]
    empty_records = sum(
        1
        for record in records
        if not any(
            [
                record.get("title"),
                record.get("abstract"),
                record.get("authors"),
                record.get("tasks"),
                record.get("datasets"),
                record.get("methods"),
                record.get("metrics"),
                record.get("implementations"),
                record.get("linked_entities"),
            ]
        )
    )
    return {
        "input_path": str(nt_path),
        "total_triples": total_triples,
        "total_papers": len(records),
        "papers_with_title": sum(1 for record in records if record.get("title")),
        "papers_with_abstract": sum(1 for record in records if record.get("abstract")),
        "empty_records": empty_records,
        "avg_title_length": (sum(title_lengths) / len(title_lengths)) if title_lengths else 0.0,
        "avg_abstract_length": (sum(abstract_lengths) / len(abstract_lengths)) if abstract_lengths else 0.0,
    }


def build_paper_records(
    nt_path: Path,
    output_path: Path,
    extraction_stats_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    require_existing_input(nt_path)
    if is_debug_input(nt_path):
        print("[build_records] sample/debug input detected; streaming with frequent progress updates.", flush=True)
    paper_map, total_triples = collect_paper_triples_pass1(nt_path)
    selected_paper_uris = sorted(paper_map)
    if limit is not None:
        selected_paper_uris = selected_paper_uris[:limit]

    referenced_nodes = sorted(
        {
            node_uri
            for paper_uri in selected_paper_uris
            for node_uri in paper_map[paper_uri]["referenced_nodes"]
        }
    )
    node_cache = collect_node_metadata_pass2(nt_path, referenced_nodes)

    records = [finalize_paper_record(paper_map[paper_uri], node_cache) for paper_uri in selected_paper_uris]
    save_jsonl(records, output_path)

    extraction_stats = compute_extraction_stats(records, nt_path, total_triples)
    save_json(extraction_stats, extraction_stats_path)
    return extraction_stats
