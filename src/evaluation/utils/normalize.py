from urllib.parse import unquote


GRAPHDB_WRAPPER = "http://localhost:7200/resource?uri="


def normalize_target_iri(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith(GRAPHDB_WRAPPER):
        value = value[len(GRAPHDB_WRAPPER):]
    return unquote(value)


def normalize_chunk_paper_id(value: str) -> str:
    if not value:
        return ""
    return unquote(value.strip())