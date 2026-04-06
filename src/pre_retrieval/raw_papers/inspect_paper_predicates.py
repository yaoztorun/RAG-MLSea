from __future__ import annotations

import argparse
from collections import Counter
import sys
from typing import Dict, List, Optional, Set

from src.pre_retrieval.config import resolve_repo_path
from src.pre_retrieval.raw_papers.build_paper_records import (
    FIELD_CANDIDATES,
    is_paper_subject,
    log_progress,
    progress_interval_for_path,
    stream_nt_triples,
)
from src.pre_retrieval.utils import save_json


def build_candidate_mappings(predicate_counter: Counter[str]) -> Dict[str, List[Dict[str, int | str]]]:
    mappings: Dict[str, List[Dict[str, int | str]]] = {}
    for field_name, predicates in FIELD_CANDIDATES.items():
        candidates = []
        for predicate in predicates:
            if predicate in predicate_counter:
                candidates.append({"predicate": predicate, "count": predicate_counter[predicate]})
        mappings[field_name] = candidates
    return mappings


def inspect_predicates_streaming(input_path, limit: Optional[int] = None) -> Dict[str, object]:
    progress_interval = progress_interval_for_path(input_path)
    predicate_counter: Counter[str] = Counter()
    paper_subjects: Set[str] = set()
    triples_processed = 0

    for triple in stream_nt_triples(input_path):
        triples_processed += 1
        if triples_processed % progress_interval == 0:
            log_progress("inspect_predicates", triples_processed, len(paper_subjects))

        subject = triple["subject"]
        if subject not in paper_subjects:
            if limit is not None and len(paper_subjects) >= limit:
                continue
            if is_paper_subject(subject, triple["predicate"], triple["object"], triple["is_literal"]):
                paper_subjects.add(subject)
            else:
                continue

        predicate_counter[triple["predicate"]] += 1

    log_progress("inspect_predicates", triples_processed, len(paper_subjects), "completed")
    return {
        "input_path": str(input_path),
        "total_triples": triples_processed,
        "total_papers": len(paper_subjects),
        "predicate_frequencies": dict(sorted(predicate_counter.items(), key=lambda item: (-item[1], item[0]))),
        "top_predicates": [
            {"predicate": predicate, "count": count}
            for predicate, count in predicate_counter.most_common(100)
        ],
        "candidate_mappings": build_candidate_mappings(predicate_counter),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect predicate usage for paper records in a local RDF dump.")
    parser.add_argument("--input-path", default="data/raw/pwc_1.nt")
    parser.add_argument("--output", default="data/intermediate/raw_papers/predicate_stats.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    input_path = resolve_repo_path(args.input_path)
    output_path = resolve_repo_path(args.output)

    try:
        payload = inspect_predicates_streaming(input_path=input_path, limit=args.limit)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    save_json(payload, output_path)
    print(f"Saved predicate stats to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
