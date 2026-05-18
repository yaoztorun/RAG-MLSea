"""
Adapter that converts retrieval-stage results.json into post-retrieval-compatible input.

Usage:
    python -m src.post_retrieval.adapters.retrieval_adapter --method pure_semantic_dense
    python -m src.post_retrieval.adapters.retrieval_adapter \
        --input  data/results/retrieval/pure_semantic_dense/results.json \
        --output data/results/post_retrieval/inputs/pure_semantic_dense_post_input.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RETRIEVAL_BASE = Path("data/results/retrieval")
_POST_INPUT_BASE = Path("data/results/post_retrieval/inputs")


# ---------------------------------------------------------------------------
# Candidate-level conversion
# ---------------------------------------------------------------------------

def _adapt_candidate(c: dict, question_id: str, missing_source_text: list[str]) -> dict:
    paper_id = c.get("normalized_entity_id") or c.get("entity_id", "")
    if not paper_id:
        print(
            f"WARNING [q={question_id}] candidate has no entity_id / normalized_entity_id",
            file=sys.stderr,
        )

    score = c.get("method_score")
    if score is None:
        score = c.get("original_score", 0.0)

    rank = c.get("new_rank")
    if rank is None:
        rank = c.get("original_rank")

    metadata = c.get("metadata") or {}
    source_text = metadata.get("source_text", "")
    if not source_text:
        missing_source_text.append(question_id)
        print(
            f"WARNING [q={question_id}] candidate '{paper_id}' has no source_text",
            file=sys.stderr,
        )

    result: dict = {
        "paper_id": paper_id,
        "score": score,
        "title": c.get("title_or_label", ""),
        "rank": rank,
        "source_text": source_text,
        "entity_type": c.get("entity_type", ""),
        "original_rank": c.get("original_rank"),
        "original_score": c.get("original_score"),
    }

    rep_src = c.get("representation_source")
    if rep_src is not None:
        result["representation_type"] = rep_src

    return result


# ---------------------------------------------------------------------------
# Question-level conversion
# ---------------------------------------------------------------------------

def _adapt_question(entry: dict, missing_source_text: list[str]) -> dict:
    question_id = entry.get("question_id", "")
    candidates = entry.get("candidates", [])

    if not candidates:
        print(
            f"WARNING [q={question_id}] has no candidates",
            file=sys.stderr,
        )

    results = [_adapt_candidate(c, question_id, missing_source_text) for c in candidates]

    return {
        "question_id": question_id,
        "question": entry.get("question", ""),
        "question_type": entry.get("question_type", ""),
        "difficulty": entry.get("difficulty"),
        "target_entity_iri": entry.get("target_entity_iri", ""),
        "expected_entity_type": entry.get("expected_entity_type", ""),
        "method_name": entry.get("method_name", ""),
        "gold_paper_id": entry.get("target_entity_iri", ""),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Payload-level conversion
# ---------------------------------------------------------------------------

def _adapt_payload(data: dict) -> tuple[dict, list[str]]:
    missing_source_text: list[str] = []
    per_question = [
        _adapt_question(entry, missing_source_text)
        for entry in data.get("per_question", [])
    ]
    output = {
        "method_name": data.get("method_name", ""),
        "source": "retrieval_adapter",
        "top_k": data.get("top_k", 10),
        "total_questions": data.get("total_questions", len(per_question)),
        "per_question": per_question,
    }
    return output, missing_source_text


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_adapter(
    method_name: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    if input_path is None:
        if method_name is None:
            raise ValueError("Provide --method or --input")
        input_path = _RETRIEVAL_BASE / method_name / "results.json"
    if output_path is None:
        if method_name is None:
            raise ValueError("Provide --method or --output")
        output_path = _POST_INPUT_BASE / f"{method_name}_post_input.json"

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    output, missing_source_text = _adapt_payload(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    total_candidates = sum(len(q["results"]) for q in output["per_question"])
    display_method = output.get("method_name") or method_name or input_path.parent.name

    print(
        f"\nAdapter complete\n"
        f"  Method              : {display_method}\n"
        f"  Questions converted : {len(output['per_question'])}\n"
        f"  Candidates converted: {total_candidates}\n"
        f"  Missing source_text : {len(missing_source_text)} warning(s)\n"
        f"  Output written to   : {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert retrieval results.json to post-retrieval input format."
    )
    parser.add_argument("--method", default=None, help="Retrieval method name (e.g. pure_semantic_dense)")
    parser.add_argument("--input", default=None, help="Override input path")
    parser.add_argument("--output", default=None, help="Override output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.method is None and (args.input is None or args.output is None):
        print("ERROR: provide --method, or both --input and --output", file=sys.stderr)
        sys.exit(1)
    run_adapter(
        method_name=args.method,
        input_path=args.input,
        output_path=args.output,
    )
