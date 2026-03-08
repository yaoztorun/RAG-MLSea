from typing import Any, Dict, List


def clean_literal(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip()


def split_authors(authors_str: str | None) -> List[str]:
    if not authors_str:
        return []
    return [a.strip() for a in authors_str.split("|") if a.strip()]


def extract_year(year_value: str | None) -> str:
    if not year_value:
        return ""
    year_value = str(year_value).strip()
    return year_value[:4] if len(year_value) >= 4 else year_value


def format_chunk_text(record: Dict[str, Any]) -> str:
    title = clean_literal(record.get("title"))
    abstract = clean_literal(record.get("abstract"))
    year = extract_year(record.get("year"))
    authors = split_authors(record.get("authors"))

    parts = []

    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {', '.join(authors)}")
    if year:
        parts.append(f"Year: {year}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    return "\n".join(parts)


def build_chunk_record(record: Dict[str, Any]) -> Dict[str, Any]:
    authors = split_authors(record.get("authors"))

    return {
        "paper_id": clean_literal(record.get("paper")),
        "title": clean_literal(record.get("title")),
        "abstract": clean_literal(record.get("abstract")),
        "year": extract_year(record.get("year")),
        "authors": authors,
        "chunk_text": format_chunk_text(record),
    }