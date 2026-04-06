from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from rdflib import Graph, Literal, URIRef

from src.pre_retrieval.utils import paper_id_from_uri, require_existing_input, save_json, save_jsonl, truncate_text, unique_preserve_order


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
FIELD_CANDIDATES: Dict[str, tuple[URIRef, ...]] = {
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


def load_graph(nt_path: Path) -> Graph:
    require_existing_input(nt_path)
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


def _first_literal(graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> Optional[str]:
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, Literal):
                value = str(obj).strip()
                return value or None
            if isinstance(obj, URIRef):
                resolved = resolve_node_text(graph, obj)
                if resolved:
                    return resolved
    return None


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
    return unique_preserve_order(local_name(obj).lower() for obj in graph.objects(node, RDF_TYPE) if isinstance(obj, URIRef))


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
    if predicate in {MLSO_HAS_RELATED_IMPLEMENTATION, SCHEMA_CODE_REPOSITORY}:
        return "implementations"
    for bucket, tokens in CATEGORY_TOKEN_MAP.items():
        if any(token in predicate_tokens or token in object_tokens for token in tokens):
            return bucket
    return None


def collect_paper_record(graph: Graph, paper: URIRef) -> Dict[str, Any]:
    paper_uri = str(paper)
    title = _first_literal(graph, paper, LABEL_PREDICATES)
    abstract = _first_literal(graph, paper, ABSTRACT_PREDICATES)

    authors: List[str] = []
    tasks: List[str] = []
    datasets: List[str] = []
    methods: List[str] = []
    metrics: List[str] = []
    implementations: List[str] = []
    linked_entities: List[Dict[str, Any]] = []
    raw_predicates: List[str] = []

    for predicate, obj in graph.predicate_objects(paper):
        predicate_str = str(predicate)
        raw_predicates.append(predicate_str)

        if predicate in AUTHOR_PREDICATES:
            authors.append(resolve_node_text(graph, obj))
            continue
        if predicate in CORE_METADATA_PREDICATES or predicate == RDF_TYPE:
            continue

        resolved_object = resolve_node_text(graph, obj)
        if not resolved_object:
            continue

        if isinstance(obj, URIRef):
            object_types = resolve_node_types(graph, obj)
            category = infer_bucket(predicate, obj, object_types)
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
                    "predicate": predicate_str,
                    "predicate_label": local_name(predicate),
                    "object_uri": str(obj),
                    "object_label": truncate_text(resolved_object, 180),
                    "object_types": object_types,
                    "category": category or "linked_entity",
                }
            )
        elif predicate == SCHEMA_CODE_REPOSITORY:
            implementations.append(resolved_object)

    record = {
        "paper_id": paper_id_from_uri(paper_uri),
        "paper_uri": paper_uri,
        "title": title,
        "abstract": abstract,
        "authors": unique_preserve_order(authors),
        "tasks": unique_preserve_order(tasks),
        "datasets": unique_preserve_order(datasets),
        "methods": unique_preserve_order(methods),
        "metrics": unique_preserve_order(metrics),
        "implementations": unique_preserve_order(implementations),
        "linked_entities": linked_entities,
        "raw_predicates": sorted(set(raw_predicates)),
    }
    return record


def compute_extraction_stats(records: List[Dict[str, Any]], nt_path: Path, graph: Graph) -> Dict[str, Any]:
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
        "total_triples": len(graph),
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
    graph = load_graph(nt_path)
    papers = iter_paper_subjects(graph)
    if limit is not None:
        papers = papers[:limit]

    records = [collect_paper_record(graph, paper) for paper in papers]
    save_jsonl(records, output_path)

    extraction_stats = compute_extraction_stats(records, nt_path, graph)
    save_json(extraction_stats, extraction_stats_path)
    return extraction_stats
