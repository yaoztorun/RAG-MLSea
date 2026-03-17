from urllib.parse import unquote


GRAPHDB_WRAPPER = "http://localhost:7200/resource?uri="


def fully_unquote(value: str) -> str:
    if not value:
        return ""
    prev = value
    current = unquote(prev)
    while current != prev:
        prev = current
        current = unquote(prev)
    return current.strip()


def normalize_target_iri(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith(GRAPHDB_WRAPPER):
        value = value[len(GRAPHDB_WRAPPER):]
    return fully_unquote(value)


def normalize_chunk_paper_id(value: str) -> str:
    if not value:
        return ""
    return fully_unquote(value.strip())