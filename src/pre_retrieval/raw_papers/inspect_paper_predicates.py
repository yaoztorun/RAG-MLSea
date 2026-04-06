from __future__ import annotations

import argparse
from collections import Counter

from src.pre_retrieval.config import resolve_repo_path
from src.pre_retrieval.io_utils import save_json
from src.pre_retrieval.raw_papers.build_paper_records import iter_paper_subjects, load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect predicate usage for paper records in a local RDF dump.")
    parser.add_argument("--nt-path", default="data/raw/pwc_1.nt")
    parser.add_argument("--output", default="data/intermediate/raw_papers/predicate_stats.json")
    args = parser.parse_args()

    nt_path = resolve_repo_path(args.nt_path)
    output_path = resolve_repo_path(args.output)

    graph = load_graph(nt_path)
    predicate_counter: Counter[str] = Counter()
    papers = iter_paper_subjects(graph)

    for paper in papers:
        for _, predicate, _ in graph.triples((paper, None, None)):
            predicate_counter[str(predicate)] += 1

    payload = {
        "nt_path": str(nt_path),
        "paper_count": len(papers),
        "graph_triple_count": len(graph),
        "top_predicates": [
            {"predicate": predicate, "count": count}
            for predicate, count in predicate_counter.most_common(100)
        ],
    }
    save_json(payload, output_path)

    print(f"Paper subjects found: {len(papers)}")
    print(f"Saved predicate stats to: {output_path}")
    for row in payload["top_predicates"][:25]:
        print(f"{row['count']:>8}  {row['predicate']}")


if __name__ == "__main__":
    main()
