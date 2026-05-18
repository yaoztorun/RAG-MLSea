"""
Batch post-retrieval stage runner.

Consumes an adapted post-retrieval input file produced by
retrieval_adapter.py and runs the existing post-retrieval pipeline
(hard filter + optional cross-encoder re-ranking + context formatting)
over all questions in the file.

Does NOT:
  - Query ChromaDB or any vector store
  - Re-run retrieval
  - Run LLM generation (unless --generate is explicitly passed)

Usage (by method name, using default paths):
  python -m src.post_retrieval.scripts.run_post_retrieval_stage \\
      --method pure_semantic_dense

Usage (explicit paths):
  python -m src.post_retrieval.scripts.run_post_retrieval_stage \\
      --input  data/results/post_retrieval/inputs/pure_semantic_dense_post_input.json \\
      --output data/results/post_retrieval/pure_semantic_dense/

RRF methods (scores ~ 0.01-0.03) — disable the hard filter:
  python -m src.post_retrieval.scripts.run_post_retrieval_stage \\
      --method optional_rrf_fusion --min-score 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Resolve repo root so relative data paths work regardless of cwd
# ---------------------------------------------------------------------------
from src.pre_retrieval.shared.config import REPO_ROOT
from src.post_retrieval.pipeline import (
    build_context_payload,
    build_paper_id_lookup,
    load_canonical_paper_records,
    load_cross_encoder,
    DEFAULT_MIN_RETRIEVAL_SCORE,
)

_POST_INPUT_BASE = REPO_ROOT / "data/results/post_retrieval/inputs"
_POST_OUTPUT_BASE = REPO_ROOT / "data/results/post_retrieval"

# ---------------------------------------------------------------------------
# RRF methods whose scores are rank-fusion values (not cosine) — must not
# apply the 0.20 cosine threshold.
# ---------------------------------------------------------------------------
_RRF_METHODS = {"optional_rrf_fusion", "optional_rrf_symbolic"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core per-question processing
# ---------------------------------------------------------------------------

def _process_question(
    entry: Dict[str, Any],
    paper_lookup: Dict[str, Any],
    *,
    cross_encoder: Any,
    use_cross_encoder: bool,
    min_retrieval_score: Optional[float],
    top_k: int,
) -> Dict[str, Any]:
    question_id = entry.get("question_id", "")
    question = entry.get("question", "")
    target_iri = entry.get("target_entity_iri", "")
    results = entry.get("results", [])

    payload = build_context_payload(
        question,
        results,
        paper_lookup,
        representation_lookup=None,   # source_text is already inline
        cross_encoder=cross_encoder,
        use_cross_encoder=use_cross_encoder,
        min_retrieval_score=min_retrieval_score,
        top_k=top_k,
    )

    # Determine whether the gold entity was in the input candidates
    input_iris = [r.get("paper_id", "") for r in results]
    gold_in_input = target_iri in input_iris

    # Determine whether the gold entity survived to the selected top-k
    selected_iris = [
        c.get("entity_iri") or c.get("paper_id", "")
        for c in payload.get("selected_results", [])
    ]
    gold_in_selected = target_iri in selected_iris

    # Build compact per-candidate records for the output
    def _compact(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for c in candidates:
            out.append({
                "paper_id": c.get("paper_id", ""),
                "entity_iri": c.get("entity_iri") or c.get("paper_id", ""),
                "title": (c.get("canonical_record") or {}).get("title") or c.get("title", ""),
                "retrieval_score": _safe_float(c.get("retrieval_score")),
                "cross_score": _safe_float(c.get("cross_score")),
                "entity_type": c.get("entity_type", ""),
                "representation_type": c.get("representation_type", ""),
                "source_text_preview": (c.get("source_text") or "")[:200],
            })
        return out

    return {
        "question_id": question_id,
        "question": question,
        "target_entity_iri": target_iri,
        "gold_in_input_candidates": gold_in_input,
        "gold_in_selected_top_k": gold_in_selected,
        "num_input_candidates": len(results),
        "num_after_filter": len(payload.get("filtered_results", [])),
        "num_selected": len(payload.get("selected_results", [])),
        "context": payload["context"],
        "input_candidates": _compact(payload.get("candidates", [])),
        "filtered_candidates": _compact(payload.get("filtered_results", [])),
        "selected_candidates": _compact(payload.get("selected_results", [])),
    }


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_stage(
    method_name: Optional[str],
    input_path: Optional[Path],
    output_dir: Optional[Path],
    *,
    min_score: Optional[float],
    top_k: int,
    skip_cross_encoder: bool,
    papers_path: str,
) -> None:
    # Resolve paths
    if input_path is None:
        if method_name is None:
            print("ERROR: provide --method or --input", file=sys.stderr)
            sys.exit(1)
        input_path = _POST_INPUT_BASE / f"{method_name}_post_input.json"
    if output_dir is None:
        resolved_method = method_name or input_path.parent.name
        output_dir = _POST_OUTPUT_BASE / resolved_method

    input_path = REPO_ROOT / input_path if not Path(input_path).is_absolute() else Path(input_path)
    output_dir = REPO_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect RRF and disable hard filter if not explicitly overridden
    detected_method = method_name or ""
    if min_score is None:
        if detected_method in _RRF_METHODS:
            effective_min_score = 0.0
            print(f"[INFO] RRF method detected — hard filter disabled (min_score=0.0)")
        else:
            effective_min_score = DEFAULT_MIN_RETRIEVAL_SCORE
            print(f"[INFO] Cosine method — hard filter active (min_score={effective_min_score})")
    else:
        effective_min_score = min_score
        print(f"[INFO] min_score explicitly set to {effective_min_score}")

    # Load data
    print(f"[1/4] Loading post-retrieval input: {input_path}")
    data = _load_json(input_path)
    per_question = data.get("per_question", [])
    display_method = data.get("method_name") or method_name or input_path.stem
    print(f"      method={display_method}, questions={len(per_question)}")

    print(f"[2/4] Loading canonical paper records from: {papers_path}")
    paper_lookup = build_paper_id_lookup(load_canonical_paper_records(papers_path))
    print(f"      Loaded {len(paper_lookup)} paper records")

    cross_encoder = None
    if not skip_cross_encoder:
        print("[3/4] Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
        cross_encoder = load_cross_encoder()
    else:
        print("[3/4] Cross-Encoder skipped (--skip-cross-encoder)")

    # Process all questions
    print(f"[4/4] Processing {len(per_question)} questions...")
    t0 = time.time()
    results_out = []
    gold_in_input_count = 0
    gold_in_selected_count = 0

    for i, entry in enumerate(per_question, start=1):
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - t0
            print(f"      [{i}/{len(per_question)}] elapsed={elapsed:.1f}s")

        result = _process_question(
            entry,
            paper_lookup,
            cross_encoder=cross_encoder,
            use_cross_encoder=not skip_cross_encoder,
            min_retrieval_score=effective_min_score if effective_min_score > 0 else None,
            top_k=top_k,
        )
        results_out.append(result)
        if result["gold_in_input_candidates"]:
            gold_in_input_count += 1
        if result["gold_in_selected_top_k"]:
            gold_in_selected_count += 1

    elapsed_total = time.time() - t0
    n = len(per_question)

    # Summary metrics
    recall_at_input = gold_in_input_count / n if n else 0.0
    recall_at_selected = gold_in_selected_count / n if n else 0.0

    summary = {
        "method_name": display_method,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_questions": n,
        "top_k": top_k,
        "min_retrieval_score": effective_min_score,
        "cross_encoder_used": not skip_cross_encoder,
        "elapsed_seconds": round(elapsed_total, 2),
        "gold_in_input_candidates": gold_in_input_count,
        "gold_in_selected_top_k": gold_in_selected_count,
        "recall_at_input": round(recall_at_input, 4),
        "recall_at_top_k_selected": round(recall_at_selected, 4),
    }

    # Write outputs
    post_results_path = output_dir / "post_results.json"
    metrics_path = output_dir / "post_metrics.json"

    _write_json(post_results_path, {
        "method_name": display_method,
        "summary": summary,
        "per_question": results_out,
    })
    _write_json(metrics_path, summary)

    print(f"\n{'='*60}")
    print(f"  Method             : {display_method}")
    print(f"  Questions          : {n}")
    print(f"  Gold in input@10   : {gold_in_input_count}/{n}  (Recall = {recall_at_input:.4f})")
    print(f"  Gold in selected@{top_k} : {gold_in_selected_count}/{n}  (Recall = {recall_at_selected:.4f})")
    print(f"  Elapsed            : {elapsed_total:.1f}s")
    print(f"  post_results.json  : {post_results_path}")
    print(f"  post_metrics.json  : {metrics_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch post-retrieval stage: filter + rerank + context assembly for all six retrieval methods."
    )
    parser.add_argument(
        "--method", default=None,
        help="Retrieval method name (e.g. pure_semantic_dense). Determines default input/output paths."
    )
    parser.add_argument("--input", default=None, help="Override input path.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    parser.add_argument(
        "--min-score", type=float, default=None,
        help=(
            "Hard similarity filter threshold. "
            "Defaults to 0.20 for cosine methods, 0.0 for RRF methods. "
            "Set explicitly to override."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidates to select (default: 3).")
    parser.add_argument(
        "--skip-cross-encoder", action="store_true",
        help="Skip cross-encoder re-ranking (uses retrieval score ordering instead)."
    )
    parser.add_argument(
        "--papers-path",
        default="data/intermediate/raw_papers/papers_master.jsonl",
        help="Path to canonical paper records JSONL."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_stage(
        method_name=args.method,
        input_path=Path(args.input) if args.input else None,
        output_dir=Path(args.output) if args.output else None,
        min_score=args.min_score,
        top_k=args.top_k,
        skip_cross_encoder=args.skip_cross_encoder,
        papers_path=args.papers_path,
    )
