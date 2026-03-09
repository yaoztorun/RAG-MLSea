import json
from pathlib import Path
from typing import Dict, Any, List


INPUT_PATH = Path("data/intermediate/chunks/papers_enriched_sample.jsonl")
OUTPUT_PATH = Path("data/intermediate/chunks/predicate_filtered_paper_sample.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def join_list(values: List[str]) -> str:
    return ", ".join(v for v in values if v)


def build_filtered_chunk(record: Dict[str, Any]) -> str:
    parts = []

    title = record.get("title", "")
    tasks = record.get("tasks", [])
    keywords = record.get("keywords", [])
    implementations = record.get("implementations", [])

    if title:
        parts.append(f"Title: {title}")
    if tasks:
        parts.append(f"Tasks: {join_list(tasks)}")
    if keywords:
        parts.append(f"Keywords: {join_list(keywords)}")
    if implementations:
        parts.append(f"Implementations: {join_list(implementations)}")

    return "\n".join(parts).strip()


def main() -> None:
    records = load_jsonl(INPUT_PATH)
    print(f"Loaded records: {len(records)}")

    output_records = []

    for record in records:
        chunk_text = build_filtered_chunk(record)
        if not chunk_text:
            continue

        new_record = dict(record)
        new_record["chunk_variant"] = "predicate_filtered_paper"
        new_record["chunk_text"] = chunk_text
        output_records.append(new_record)

    save_jsonl(output_records, OUTPUT_PATH)
    print(f"Saved {len(output_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()