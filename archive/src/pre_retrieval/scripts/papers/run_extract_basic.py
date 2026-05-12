import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.pre_retrieval.chunking.chunk_formatter import build_chunk_record
from src.pre_retrieval.raw_papers.extract_papers_from_nt import (
    extract_basic_rows_from_graph,
    load_graph,
)

NT_PATH = Path(os.getenv("MLSEA_NT_PATH", "data/raw/pwc_1.nt"))

OUTPUT_PATH = Path("data/intermediate/chunks/papers/papers_basic_sample.jsonl")


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print(f"Loading graph from NT file: {NT_PATH}")
    print("Sample limit: full dataset")

    graph = load_graph(NT_PATH)
    rows = extract_basic_rows_from_graph(graph, limit=None)

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