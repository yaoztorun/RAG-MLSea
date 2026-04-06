from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from rdflib import Graph, Literal, URIRef

from src.pre_retrieval.io_utils import save_json, save_jsonl, truncate_text, unique_preserve_order


RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
FOAF_NAME = URIRef("http://xmlns.com/foaf/0.1/name")
DCTERMS_TITLE = URIRef("http://purl.org/dc/terms/title")
DCTERMS_CREATOR = URIRef("http://purl.org/dc/terms/creator")
DCTERMS_ISSUED = URIRef("http://purl.org/dc/terms/issued")
FABIO_ABSTRACT = URIRef("http://purl.org/spar/fabio/abstract")
DCAT_KEYWORD = URIRef("http://www.w3.org/ns/dcat#keyword")
SCHEMA_NAME = URIRef("https://schema.org/name")
SCHEMA_DESCRIPTION = URIRef("https://schema.org/description")
SCHEMA_DATE_PUBLISHED = URIRef("https://schema.org/datePublished")
SCHEMA_AUTHOR = URIRef("https://schema.org/author")
SCHEMA_CODE_REPOSITORY = URIRef("https://schema.org/codeRepository")

MLSO_SCIENTIFIC_WORK = URIRef("http://w3id.org/mlso/ScientificWork")
MLSO_HAS_TASK_TYPE = URIRef("http://w3id.org/mlso/hasTaskType")
MLSO_HAS_RELATED_IMPLEMENTATION = URIRef("http://w3id.org/mlso/hasRelatedImplementation")

PAPER_PREFIX = "http://w3id.org/mlsea/pwc/scientificWork/"
LABEL_PREDICATES = (DCTERMS_TITLE, RDFS_LABEL, FOAF_NAME, SCHEMA_NAME)
ABSTRACT_PREDICATES = (FABIO_ABSTRACT, SCHEMA_DESCRIPTION)
YEAR_PREDICATES = (DCTERMS_ISSUED, SCHEMA_DATE_PUBLISHED)
AUTHOR_PREDICATES = (DCTERMS_CREATOR, SCHEMA_AUTHOR)
KEYWORD_PREDICATES = (DCAT_KEYWORD,)

CATEGORY_TOKEN_MAP = {
    "tasks": {"task"},
    "datasets": {"dataset", "benchmark", "corpus"},
    "methods": {"method", "model", "architecture", "algorithm", "approach"},
    "metrics": {"metric", "measure", "score"},
    "implementations": {"implementation", "repository", "github", "code"},
}
CORE_METADATA_PREDICATES = set(LABEL_PREDICATES + ABSTRACT_PREDICATES + YEAR_PREDICATES + AUTHOR_PREDICATES + KEYWORD_PREDICATES)


def load_graph(nt_path: Path) -> Graph:
    if not nt_path.exists():
        raise FileNotFoundError(f"NT file not found: {nt_path.resolve()}")

    graph = Graph()
    graph.parse(str(nt_path), format="nt")
    return graph


def local_name(uri: URIRef) -> str:
    value = str(uri)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    return value


def extract_year(value: str) -> str:
    cleaned = value.strip()
    return cleaned[:4] if len(cleaned) >= 4 else cleaned


def _first_literal(graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> str:
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, Literal):
                return str(obj).strip()
            if isinstance(obj, URIRef):
                resolved = resolve_node_text(graph, obj)
                if resolved:
                    return resolved
    return ""


def resolve_node_text(graph: Graph, node: URIRef | Literal | Any) -> str:
    if isinstance(node, Literal):
        return str(node).strip()
    if not isinstance(node, URIRef):
        return str(node).strip()

    for predicate in LABEL_PREDICATES:
        for obj in graph.objects(node, predicate):
            if isinstance(obj, Literal):
                return str(obj).strip()

    return local_name(node).replace("_", " ").strip()


def resolve_node_types(graph: Graph, node: URIRef) -> List[str]:
    types = [local_name(obj).lower() for obj in graph.objects(node, RDF_TYPE) if isinstance(obj, URIRef)]
    return unique_preserve_order(types)


def iter_paper_subjects(graph: Graph) -> List[URIRef]:
    subjects: Set[URIRef] = {
        subject
        for subject in graph.subjects(RDF_TYPE, MLSO_SCIENTIFIC_WORK)
        if isinstance(subject, URIRef)
    }

    subjects.update(
        subject
        for subject in graph.subjects()
        if isinstance(subject, URIRef) and str(subject).startswith(PAPER_PREFIX)
    )
    return sorted(subjects, key=str)


def infer_bucket(predicate: URIRef, object_node: URIRef, object_types: List[str]) -> Optional[str]:
    predicate_tokens = local_name(predicate).lower()
    object_tokens = f"{local_name(object_node).lower()} {' '.join(object_types)}"

    if predicate == MLSO_HAS_TASK_TYPE:
        return "tasks"
    if predicate == MLSO_HAS_RELATED_IMPLEMENTATION or predicate == SCHEMA_CODE_REPOSITORY:
        return "implementations"

    for bucket, tokens in CATEGORY_TOKEN_MAP.items():
        if any(token in predicate_tokens or token in object_tokens for token in tokens):
            return bucket
    return None


def collect_paper_record(graph: Graph, paper: URIRef) -> Dict[str, Any]:
    title = _first_literal(graph, paper, LABEL_PREDICATES)
    abstract = _first_literal(graph, paper, ABSTRACT_PREDICATES)
    year = extract_year(_first_literal(graph, paper, YEAR_PREDICATES))

    authors: List[str] = []
    keywords: List[str] = []
    tasks: List[str] = []
    datasets: List[str] = []
    methods: List[str] = []
    metrics: List[str] = []
    implementations: List[str] = []
    linked_entities: List[Dict[str, str]] = []
    predicate_counter: Counter[str] = Counter()

    for predicate, obj in graph.predicate_objects(paper):
        predicate_counter[str(predicate)] += 1

        if predicate in AUTHOR_PREDICATES:
            authors.append(resolve_node_text(graph, obj))
            continue

        if predicate in KEYWORD_PREDICATES:
            keywords.append(resolve_node_text(graph, obj))
            continue

        if predicate in CORE_METADATA_PREDICATES or predicate == RDF_TYPE:
            continue

        resolved_object = resolve_node_text(graph, obj)
        if not resolved_object:
            continue

        if isinstance(obj, URIRef):
            object_types = resolve_node_types(graph, obj)
            bucket = infer_bucket(predicate, obj, object_types)
            if bucket == "tasks":
                tasks.append(resolved_object)
            elif bucket == "datasets":
                datasets.append(resolved_object)
            elif bucket == "methods":
                methods.append(resolved_object)
            elif bucket == "metrics":
                metrics.append(resolved_object)
            elif bucket == "implementations":
                implementations.append(resolved_object)

            linked_entities.append(
                {
                    "predicate": str(predicate),
                    "predicate_label": local_name(predicate),
                    "object_id": str(obj),
                    "object_label": truncate_text(resolved_object, 180),
                    "object_types": ", ".join(object_types),
                    "category": bucket or "linked_entity",
                }
            )
        elif predicate == SCHEMA_CODE_REPOSITORY:
            implementations.append(resolved_object)

    authors = unique_preserve_order(authors)
    keywords = unique_preserve_order(keywords)
    tasks = unique_preserve_order(tasks)
    datasets = unique_preserve_order(datasets)
    methods = unique_preserve_order(methods)
    metrics = unique_preserve_order(metrics)
    implementations = unique_preserve_order(implementations)

    return {
        "paper_id": str(paper),
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "tasks": tasks,
        "datasets": datasets,
        "methods": methods,
        "metrics": metrics,
        "implementations": implementations,
        "keywords": keywords,
        "linked_entities": linked_entities,
        "field_stats": {
            "author_count": len(authors),
            "task_count": len(tasks),
            "dataset_count": len(datasets),
            "method_count": len(methods),
            "metric_count": len(metrics),
            "implementation_count": len(implementations),
            "keyword_count": len(keywords),
            "linked_entity_count": len(linked_entities),
            "title_length": len(title),
            "abstract_length": len(abstract),
        },
        "predicate_counts": dict(sorted(predicate_counter.items())),
    }


def build_paper_records(
    nt_path: Path,
    output_path: Path,
    predicate_stats_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    graph = load_graph(nt_path)
    subjects = iter_paper_subjects(graph)
    if limit is not None:
        subjects = subjects[:limit]

    records = [collect_paper_record(graph, paper) for paper in subjects]
    save_jsonl(records, output_path)

    overall_predicates: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    for record in records:
        overall_predicates.update(record.get("predicate_counts", {}))
        for field in ("title", "abstract", "authors", "tasks", "datasets", "methods", "metrics", "implementations", "keywords"):
            value = record.get(field)
            if value:
                coverage[field] += 1

    summary = {
        "nt_path": str(nt_path),
        "paper_count": len(records),
        "graph_triple_count": len(graph),
        "field_coverage": dict(sorted(coverage.items())),
        "top_predicates": [
            {"predicate": predicate, "count": count}
            for predicate, count in overall_predicates.most_common(50)
        ],
    }
    if predicate_stats_path is not None:
        save_json(summary, predicate_stats_path)
    return summary
