from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from rdflib import Graph, URIRef


RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
FOAF_NAME = URIRef("http://xmlns.com/foaf/0.1/name")

MLSO_SCIENTIFIC_WORK = URIRef("http://w3id.org/mlso/ScientificWork")
MLSO_HAS_TASK_TYPE = URIRef("http://w3id.org/mlso/hasTaskType")
MLSO_HAS_RELATED_IMPLEMENTATION = URIRef("http://w3id.org/mlso/hasRelatedImplementation")

DCTERMS_TITLE = URIRef("http://purl.org/dc/terms/title")
DCTERMS_ISSUED = URIRef("http://purl.org/dc/terms/issued")
DCTERMS_CREATOR = URIRef("http://purl.org/dc/terms/creator")

FABIO_ABSTRACT = URIRef("http://purl.org/spar/fabio/abstract")
DCAT_KEYWORD = URIRef("http://www.w3.org/ns/dcat#keyword")
SCHEMA_CODE_REPOSITORY = URIRef("https://schema.org/codeRepository")


def load_graph(nt_path: Path) -> Graph:
    if not nt_path.exists():
        raise FileNotFoundError(f"NT file not found: {nt_path.resolve()}")

    graph = Graph()
    graph.parse(str(nt_path), format="nt")
    return graph


def _to_local_name(uri: URIRef) -> str:
    value = str(uri)
    if "#" in value:
        return value.rsplit("#", 1)[-1].strip()
    if "/" in value:
        return value.rsplit("/", 1)[-1].strip()
    return value.strip()


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _first_object_literal(graph: Graph, subject: URIRef, predicate: URIRef) -> str:
    for obj in graph.objects(subject, predicate):
        return str(obj).strip()
    return ""


def _extract_year(issued_value: str) -> str:
    issued_value = issued_value.strip()
    if len(issued_value) >= 4:
        return issued_value[:4]
    return issued_value


def _collect_author_names(graph: Graph, paper: URIRef) -> List[str]:
    names: List[str] = []
    for author in graph.objects(paper, DCTERMS_CREATOR):
        if not isinstance(author, URIRef):
            names.append(str(author))
            continue

        name = _first_object_literal(graph, author, FOAF_NAME)
        if not name:
            name = _first_object_literal(graph, author, RDFS_LABEL)
        if not name:
            name = _to_local_name(author)

        names.append(name)

    return _unique_preserve_order(names)


def _collect_task_values(graph: Graph, paper: URIRef) -> List[str]:
    values: List[str] = []
    for task in graph.objects(paper, MLSO_HAS_TASK_TYPE):
        if not isinstance(task, URIRef):
            values.append(str(task))
            continue

        task_name = _first_object_literal(graph, task, RDFS_LABEL)
        if not task_name:
            task_name = _first_object_literal(graph, task, FOAF_NAME)
        if not task_name:
            task_name = _to_local_name(task)

        values.append(task_name)

    return _unique_preserve_order(values)


def _collect_keyword_values(graph: Graph, paper: URIRef) -> List[str]:
    return _unique_preserve_order(str(keyword) for keyword in graph.objects(paper, DCAT_KEYWORD))


def _collect_implementation_values(graph: Graph, paper: URIRef) -> List[str]:
    values: List[str] = []
    for implementation in graph.objects(paper, MLSO_HAS_RELATED_IMPLEMENTATION):
        if not isinstance(implementation, URIRef):
            values.append(str(implementation))
            continue

        impl_value = _first_object_literal(graph, implementation, RDFS_LABEL)
        if not impl_value:
            impl_value = _first_object_literal(graph, implementation, SCHEMA_CODE_REPOSITORY)
        if not impl_value:
            impl_value = _to_local_name(implementation)

        values.append(impl_value)

    return _unique_preserve_order(values)


def _iter_scientific_work_subjects(graph: Graph) -> List[URIRef]:
    subjects = [
        subject
        for subject in graph.subjects(RDF_TYPE, MLSO_SCIENTIFIC_WORK)
        if isinstance(subject, URIRef)
    ]
    return sorted(set(subjects), key=str)


def extract_basic_rows_from_graph(graph: Graph, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for paper in _iter_scientific_work_subjects(graph):
        title = _first_object_literal(graph, paper, DCTERMS_TITLE)
        year = _extract_year(_first_object_literal(graph, paper, DCTERMS_ISSUED))
        authors = _collect_author_names(graph, paper)

        rows.append(
            {
                "paper": str(paper),
                "title": title,
                "year": year,
                "authors": authors,
            }
        )

        if limit is not None and len(rows) >= limit:
            break

    return rows


def extract_enriched_rows_from_graph(graph: Graph, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for paper in _iter_scientific_work_subjects(graph):
        title = _first_object_literal(graph, paper, DCTERMS_TITLE)
        abstract = _first_object_literal(graph, paper, FABIO_ABSTRACT)
        year = _extract_year(_first_object_literal(graph, paper, DCTERMS_ISSUED))

        rows.append(
            {
                "paper": str(paper),
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": _collect_author_names(graph, paper),
                "tasks": _collect_task_values(graph, paper),
                "keywords": _collect_keyword_values(graph, paper),
                "implementations": _collect_implementation_values(graph, paper),
            }
        )

        if limit is not None and len(rows) >= limit:
            break

    return rows
