import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests

from src.pre_retrieval.chunking.chunk_formatter import build_chunk_record


GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/MLSea_Thesis"
)

QUERY_PATH = Path("src/pre_retrieval/sparql/extract_papers_basic.rq")
OUTPUT_PATH = Path("data/intermediate/chunks/papers/papers_basic_sample.jsonl")


def load_query(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_sparql_query(endpoint: str, query: str) -> Dict[str, Any]:
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.post(
        endpoint,
        data={"query": query},
        headers=headers,
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def parse_bindings(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    bindings = results.get("results", {}).get("bindings", [])
    parsed = []

    for row in bindings:
        parsed.append({
            "paper": row.get("paper", {}).get("value", ""),
            "title": row.get("title", {}).get("value", ""),
            "abstract": row.get("abstract", {}).get("value", ""),
            "year": row.get("year", {}).get("value", ""),
            "authors": row.get("authors", {}).get("value", ""),
        })

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
    rows = parse_bindings(results)

    print(f"Rows returned: {len(rows)}")

    chunk_records = []
    for row in rows:
        chunk = build_chunk_record(row)
        if chunk["paper_id"] and chunk["chunk_text"]:
            chunk_records.append(chunk)

    print(f"Chunk records built: {len(chunk_records)}")

    save_jsonl(chunk_records, OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()