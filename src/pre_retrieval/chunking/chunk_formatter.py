from typing import Any, Dict, List


def clean_literal(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip()


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


def extract_year(year_value: str | None) -> str:
    if not year_value:
        return ""
    year_value = str(year_value).strip()
    return year_value[:4] if len(year_value) >= 4 else year_value


def format_chunk_text(record: Dict[str, Any]) -> str:
    title = clean_literal(record.get("title"))
    abstract = clean_literal(record.get("abstract"))
    year = extract_year(record.get("year"))
    authors = split_pipe_values(record.get("authors"))
    tasks = split_pipe_values(record.get("tasks"))
    keywords = split_pipe_values(record.get("keywords"))
    implementations = split_pipe_values(record.get("implementations"))

    parts = []

    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {', '.join(authors)}")
    if year:
        parts.append(f"Year: {year}")
    if tasks:
        parts.append(f"Tasks: {', '.join(tasks)}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if implementations:
        parts.append(f"Implementations: {', '.join(implementations)}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    return "\n".join(parts)


def build_chunk_record(record: Dict[str, Any]) -> Dict[str, Any]:
    authors = split_pipe_values(record.get("authors"))
    tasks = split_pipe_values(record.get("tasks"))
    keywords = split_pipe_values(record.get("keywords"))
    implementations = split_pipe_values(record.get("implementations"))

    return {
        "paper_id": clean_literal(record.get("paper")),
        "title": clean_literal(record.get("title")),
        "abstract": clean_literal(record.get("abstract")),
        "year": extract_year(record.get("year")),
        "authors": authors,
        "tasks": tasks,
        "keywords": keywords,
        "implementations": implementations,
        "chunk_text": format_chunk_text(record),
    }