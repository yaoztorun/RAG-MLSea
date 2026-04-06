from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef


NT_PATH = Path("data/raw/pwc_1.nt")
PAPER_PREFIX = "http://w3id.org/mlsea/pwc/scientificWork/"


def main() -> None:
    print(f"Loading graph from: {NT_PATH}")

    if not NT_PATH.exists():
        raise FileNotFoundError(f"File not found: {NT_PATH.resolve()}")

    g = Graph()
    g.parse(str(NT_PATH), format="nt")

    print(f"Total triples loaded: {len(g)}")

    predicate_counter = Counter()
    paper_subjects = set()

    for s, p, o in g:
        if isinstance(s, URIRef) and str(s).startswith(PAPER_PREFIX):
            paper_subjects.add(str(s))
            predicate_counter[str(p)] += 1

    print(f"\nPaper subjects found: {len(paper_subjects)}")
    print("\nTop predicates used by paper subjects:\n")

    for predicate, count in predicate_counter.most_common(50):
        print(f"{count:>8}  {predicate}")


if __name__ == "__main__":
    main()