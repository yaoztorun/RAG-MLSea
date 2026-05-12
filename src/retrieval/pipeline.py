from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retrieval.aggregate_retrieval_stage import aggregate_all_methods, write_json
from src.retrieval.config import (
    ALL_METHODS,
    DEFAULT_TOP_K,
    METHOD_ALIASES,
    RETRIEVAL_OUTPUT_DIR,
)
from src.retrieval.data_loading import load_questions
from src.retrieval.evaluate_retrieval_stage import evaluate_results


def _method_output_dir(method_name: str, output_dir: Path) -> Path:
    return output_dir / method_name


def _results_path(method_name: str, output_dir: Path) -> Path:
    return _method_output_dir(method_name, output_dir) / "results.json"


def _metrics_path(method_name: str, output_dir: Path) -> Path:
    return _method_output_dir(method_name, output_dir) / "metrics.json"


def _already_done(method_name: str, output_dir: Path) -> bool:
    return (
        _results_path(method_name, output_dir).exists()
        and _metrics_path(method_name, output_dir).exists()
    )


def _run_method_impl(
    method_name: str,
    questions: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if method_name == "pure_semantic_dense":
        from src.retrieval.dense_baseline import run_dense_baseline
        return run_dense_baseline(questions=questions, top_k=top_k)
    if method_name == "hybrid_type_filtering":
        from src.retrieval.filtering import run_type_filtering
        return run_type_filtering(questions=questions, top_k=top_k)
    if method_name == "hybrid_type_onehop_filtering":
        from src.retrieval.filtering import run_hybrid_type_onehop_filtering
        return run_hybrid_type_onehop_filtering(questions=questions, top_k=top_k)
    if method_name == "hybrid_predicate_aware_filtering":
        from src.retrieval.filtering import run_predicate_aware_filtering
        return run_predicate_aware_filtering(questions=questions, top_k=top_k)
    if method_name == "optional_rrf_fusion":
        from src.retrieval.rrf import run_rrf_fusion
        return run_rrf_fusion(questions=questions, top_k=top_k)
    if method_name == "optional_rrf_symbolic":
        from src.retrieval.rrf import run_rrf_symbolic
        return run_rrf_symbolic(questions=questions, top_k=top_k)
    raise ValueError(f"Unknown method: {method_name!r}. Valid methods: {ALL_METHODS}")


def run_method(
    method_name: str,
    questions: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    output_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = RETRIEVAL_OUTPUT_DIR

    # Resolve backward-compatibility alias
    method_name = METHOD_ALIASES.get(method_name, method_name)

    if not force and _already_done(method_name, output_dir):
        if verbose:
            print(f"[{method_name}] already done, skipping (use --force to rerun)")
        rp = _results_path(method_name, output_dir)
        with rp.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload

    if questions is None:
        questions = load_questions()

    if verbose:
        print(f"[{method_name}] running...")

    results = _run_method_impl(method_name, questions, top_k)
    metrics = evaluate_results(results)

    results_payload = {
        "method_name": method_name,
        "top_k": top_k,
        "total_questions": len(results),
        "per_question": results,
    }
    metrics_payload = {
        "method_name": method_name,
        "top_k": top_k,
        **metrics,
    }

    results_p = _results_path(method_name, output_dir)
    metrics_p = _metrics_path(method_name, output_dir)
    write_json(results_payload, results_p)
    write_json(metrics_payload, metrics_p)

    if verbose:
        overall = metrics.get("overall", {})
        ndcg = overall.get("NDCG", 0.0)
        hit1 = overall.get("Hit@1", 0.0)
        n = overall.get("count", 0)
        print(f"[{method_name}] done — n={n}, Hit@1={hit1:.4f}, NDCG={ndcg:.4f}")
        print(f"  saved -> {results_p}")
        print(f"  saved -> {metrics_p}")

    return results_payload


def run_all_methods(
    top_k: int = DEFAULT_TOP_K,
    output_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> None:
    if output_dir is None:
        output_dir = RETRIEVAL_OUTPUT_DIR

    questions = load_questions()

    for method_name in ALL_METHODS:
        run_method(
            method_name=method_name,
            questions=questions,
            top_k=top_k,
            output_dir=output_dir,
            force=force,
            verbose=verbose,
        )

    if verbose:
        print("\nAggregating summaries...")
    aggregate_all_methods(output_dir=output_dir)
    if verbose:
        print(f"Summaries written to {output_dir}")
