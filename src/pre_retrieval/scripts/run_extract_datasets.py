import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/MLSea_Thesis"
)

QUERY_PATH = Path("src/pre_retrieval/sparql/extract_datasets_basic.rq")
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

    items = []
    seen = set()

    for part in value.split("|"):
        cleaned = part.strip()

        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)

    return items


def build_chunk_text(record: Dict[str, Any]) -> str:
    parts = []

    if record.get("dataset_label"):
        parts.append(f"Dataset: {record['dataset_label']}")

    if record.get("tasks"):
        parts.append(f"Tasks: {', '.join(record['tasks'])}")

    if record.get("scientific_references"):
        parts.append(f"Scientific References: {', '.join(record['scientific_references'])}")

    if record.get("sources"):
        parts.append("Sources:")
        parts.extend(record["sources"])

    if record.get("loaders"):
        parts.append("Data Loader Locations:")
        parts.extend(record["loaders"])

    if record.get("variants"):
        parts.append(f"Variants: {', '.join(record['variants'])}")

    if record.get("is_variant_of"):
        parts.append(f"Is Variant Of: {', '.join(record['is_variant_of'])}")

    return "\n".join(parts)


def parse_bindings(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    bindings = results.get("results", {}).get("bindings", [])
    parsed = []

    for row in bindings:
        record = {
            "dataset_id": row.get("dataset", {}).get("value", ""),
            "dataset_label": row.get("dataset_label", {}).get("value", ""),
            "tasks": split_pipe_values(row.get("tasks", {}).get("value", "")),
            "loaders": split_pipe_values(row.get("loaders", {}).get("value", "")),
            "sources": split_pipe_values(row.get("sources", {}).get("value", "")),
            "scientific_references": split_pipe_values(row.get("scientific_references", {}).get("value", "")),
            "variants": split_pipe_values(row.get("variants", {}).get("value", "")),
            "is_variant_of": split_pipe_values(row.get("is_variant_of", {}).get("value", "")),
        }

        record["chunk_text"] = build_chunk_text(record)

        if record["dataset_id"] and record["chunk_text"]:
            parsed.append(record)

    return parsed


def save_jsonl(records: List[Dict[str, Any]], output_path: Path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

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