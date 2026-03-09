import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/MLSea_Thesis"
)

QUERY_PATH = Path("src/pre_retrieval/sparql/extract_dataset_chunks.rq")
OUTPUT_PATH = Path("data/intermediate/chunks/dataset_chunks_sample.jsonl")


def load_query(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_sparql_query(endpoint: str, query: str) -> Dict[str, Any]:
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.post(
        endpoint,
        data={"query": query},
        headers=headers,
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def split_pipe_values(value: str | None) -> List[str]:
    if not value:
        return []
    seen = set()
    items = []
    for part in value.split("|"):
        cleaned = part.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items


def build_chunk_text(record: Dict[str, Any]) -> str:
    parts = []

    label = record.get("dataset_label", "").strip()
    tasks = split_pipe_values(record.get("tasks", ""))
    papers = split_pipe_values(record.get("papers", ""))
    keywords = split_pipe_values(record.get("keywords", ""))

    if label:
        parts.append(f"Dataset: {label}")
    if tasks:
        parts.append(f"Tasks: {', '.join(tasks)}")
    if papers:
        parts.append(f"Related Papers: {', '.join(papers)}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return "\n".join(parts)


def parse_bindings(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    bindings = results.get("results", {}).get("bindings", [])
    parsed = []

    for row in bindings:
        record = {
            "dataset_id": row.get("dataset", {}).get("value", ""),
            "dataset_label": row.get("dataset_label", {}).get("value", ""),
            "tasks": split_pipe_values(row.get("tasks", {}).get("value", "")),
            "papers": split_pipe_values(row.get("papers", {}).get("value", "")),
            "keywords": split_pipe_values(row.get("keywords", {}).get("value", "")),
        }
        record["chunk_text"] = build_chunk_text({
            "dataset_label": record["dataset_label"],
            "tasks": " | ".join(record["tasks"]),
            "papers": " | ".join(record["papers"]),
            "keywords": " | ".join(record["keywords"]),
        })
        parsed.append(record)

    return parsed


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print(f"Using GraphDB endpoint: {GRAPHDB_ENDPOINT}")
    print(f"Loading query from: {QUERY_PATH}")

    query = load_query(QUERY_PATH)
    results = run_sparql_query(GRAPHDB_ENDPOINT, query)
    records = parse_bindings(results)

    print(f"Dataset chunk records built: {len(records)}")
    save_jsonl(records, OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()