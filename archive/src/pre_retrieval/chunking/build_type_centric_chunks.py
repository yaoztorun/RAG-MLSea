import json
from pathlib import Path
from typing import Dict, Any, List


INPUT_PATH = Path("data/intermediate/chunks/task_chunks_sample.jsonl")
OUTPUT_PATH = Path("data/intermediate/chunks/type_centric_task_sample.jsonl")


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


def build_type_centric_chunk(record: Dict[str, Any]) -> str:
    parts = []

    task_label = record.get("task_label", "")
    papers = record.get("papers", [])
    keywords = record.get("keywords", [])
    implementations = record.get("implementations", [])

    if task_label:
        parts.append(f"Task Type: {task_label}")
    if papers:
        parts.append(f"Representative Papers: {', '.join(papers)}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if implementations:
        parts.append(f"Implementations: {', '.join(implementations)}")

    return "\n".join(parts).strip()


def main() -> None:
    records = load_jsonl(INPUT_PATH)
    print(f"Loaded records: {len(records)}")

    output_records = []

    for record in records:
        chunk_text = build_type_centric_chunk(record)
        if not chunk_text:
            continue

        new_record = dict(record)
        new_record["chunk_variant"] = "type_centric_task"
        new_record["chunk_text"] = chunk_text
        output_records.append(new_record)

    save_jsonl(output_records, OUTPUT_PATH)
    print(f"Saved {len(output_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()