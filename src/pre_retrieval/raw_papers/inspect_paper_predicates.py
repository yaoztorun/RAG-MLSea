from __future__ import annotations

import argparse
from collections import Counter
import sys
from typing import Dict, List

from src.pre_retrieval.config import resolve_repo_path
from src.pre_retrieval.raw_papers.build_paper_records import FIELD_CANDIDATES, iter_paper_subjects, load_graph
from src.pre_retrieval.utils import save_json


def build_candidate_mappings(predicate_counter: Counter[str]) -> Dict[str, List[Dict[str, int | str]]]:
    mappings: Dict[str, List[Dict[str, int | str]]] = {}
    for field_name, predicates in FIELD_CANDIDATES.items():
        candidates = []
        for predicate in predicates:
            predicate_str = str(predicate)
            if predicate_str in predicate_counter:
                candidates.append({"predicate": predicate_str, "count": predicate_counter[predicate_str]})
        mappings[field_name] = candidates
    return mappings


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect predicate usage for paper records in a local RDF dump.")
    parser.add_argument("--input-path", default="data/raw/pwc_1.nt")
    parser.add_argument("--output", default="data/intermediate/raw_papers/predicate_stats.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    del args.force_rebuild
    input_path = resolve_repo_path(args.input_path)
    output_path = resolve_repo_path(args.output)

    try:
        graph = load_graph(input_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    papers = iter_paper_subjects(graph)
    if args.limit is not None:
        papers = papers[: args.limit]

    predicate_counter: Counter[str] = Counter()
    for paper in papers:
        for _, predicate, _ in graph.triples((paper, None, None)):
            predicate_counter[str(predicate)] += 1

    payload = {
        "input_path": str(input_path),
        "total_triples": len(graph),
        "total_papers": len(papers),
        "predicate_frequencies": dict(sorted(predicate_counter.items(), key=lambda item: (-item[1], item[0]))),
        "top_predicates": [
            {"predicate": predicate, "count": count}
            for predicate, count in predicate_counter.most_common(100)
        ],
        "candidate_mappings": build_candidate_mappings(predicate_counter),
    }
    save_json(payload, output_path)
    print(f"Saved predicate stats to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
