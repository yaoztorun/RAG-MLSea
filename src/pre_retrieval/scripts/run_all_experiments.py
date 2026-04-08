from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.pre_retrieval.chunking.build_representations import build_representations
from src.pre_retrieval.config import default_subset_records_path, default_subset_stats_path, load_pipeline_config, resolve_repo_path
from src.pre_retrieval.evaluation.aggregate_results import aggregate_result_files
from src.pre_retrieval.evaluation.evaluate_retrieval import evaluate_representation, representation_results_path
from src.pre_retrieval.embeddings.embed_and_store import embed_and_store_representations
from src.pre_retrieval.raw_papers.build_curated_subset import build_curated_subset
from src.pre_retrieval.utils import require_existing_input


EXPERIMENT_REPRESENTATIONS = [
    "title_only",
    "abstract_only",
    "title_abstract",
    "enriched_metadata",
    "predicate_filtered",
    "one_hop",
]


def _should_skip_representation(result_path: Path, skip_existing: bool, force_rebuild: bool) -> bool:
    return skip_existing and not force_rebuild and result_path.exists()


def _resolve_records_path(
    config: Dict[str, Any],
    *,
    records_path: str | None,
    papers_path: Path,
    questions_path: Path,
    disable_subset: bool,
    max_papers: int | None,
) -> Path:
    if records_path:
        resolved = resolve_repo_path(records_path)
        require_existing_input(resolved)
        return resolved

    subset_config = config.get("corpus_subset", {})
    use_subset = bool(subset_config.get("enabled", False)) and not disable_subset
    if not use_subset:
        require_existing_input(papers_path)
        return papers_path

    subset_max_papers = int(max_papers or subset_config.get("max_papers", 200000))
    subset_output_path = resolve_repo_path(default_subset_records_path(subset_max_papers))
    subset_stats_path = resolve_repo_path(default_subset_stats_path())
    build_curated_subset(
        papers_master_path=papers_path,
        questions_path=questions_path,
        output_path=subset_output_path,
        stats_output_path=subset_stats_path,
        max_papers=subset_max_papers,
        include_gold_targets=bool(subset_config.get("include_gold_targets", True)),
    )
    return subset_output_path


def _maybe_build_representation(
    representation: str,
    records_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    limit: int | None,
    skip_existing: bool,
    force_rebuild: bool,
) -> str:
    representation_path = output_dir / f"{representation}.jsonl"
    if skip_existing and not force_rebuild and representation_path.exists():
        return "skipped existing representation file"

    counts = build_representations(
        records_path=records_path,
        output_dir=output_dir,
        representation_types=[representation],
        representation_config_map=config["representations"],
        limit=limit,
    )
    return f"built {counts.get(representation, 0)} records"


def _run_representation(
    representation: str,
    config: Dict[str, Any],
    records_path: Path,
    representation_dir: Path,
    questions_path: Path,
    results_dir: Path,
    limit: int | None,
    skip_existing: bool,
    force_rebuild: bool,
) -> None:
    result_path = representation_results_path(results_dir, representation)
    print(f"[{representation}] starting")
    if _should_skip_representation(result_path, skip_existing=skip_existing, force_rebuild=force_rebuild):
        print(f"[{representation}] skipped existing evaluation output")
        return

    build_status = _maybe_build_representation(
        representation=representation,
        records_path=records_path,
        output_dir=representation_dir,
        config=config,
        limit=limit,
        skip_existing=skip_existing,
        force_rebuild=force_rebuild,
    )
    print(f"[{representation}] build status: {build_status}")

    embed_summary = embed_and_store_representations(
        representation_path=representation_dir / f"{representation}.jsonl",
        vector_store_config=config["vector_store"],
        representation_type=representation,
        embedder_type=config["embedder_type"],
        model_name=config["model_name"],
        force_rebuild=force_rebuild,
        limit=limit,
    )
    print(
        f"[{representation}] embedding status: inserted={embed_summary['inserted_count']} skipped={embed_summary['skipped_count']} "
        f"collection_size={embed_summary['collection_size']}"
    )

    evaluate_representation(
        representation_type=representation,
        questions_path=questions_path,
        records_path=records_path,
        vector_store_config=config["vector_store"],
        embedder_type=config["embedder_type"],
        model_name=config["model_name"],
        top_k_values=config["evaluation"]["top_k"],
        output_path=result_path,
        limit=limit,
        representation_order=config["evaluation"]["representation_order"],
    )
    print(f"[{representation}] evaluation complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the active local pre-retrieval experiments and aggregate results.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--papers-path", default="data/intermediate/raw_papers/papers_master.jsonl")
    parser.add_argument("--records-path", default=None)
    parser.add_argument("--representation-dir", default="data/intermediate/representations")
    parser.add_argument("--questions-path", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--disable-subset", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    papers_path = resolve_repo_path(args.papers_path)
    representation_dir = resolve_repo_path(args.representation_dir)
    questions_path = resolve_repo_path(args.questions_path or config["evaluation"]["questions_path"])
    results_dir = resolve_repo_path(args.results_dir or config["evaluation"]["output_dir"])

    require_existing_input(papers_path)
    require_existing_input(questions_path)
    records_path = _resolve_records_path(
        config,
        records_path=args.records_path,
        papers_path=papers_path,
        questions_path=questions_path,
        disable_subset=args.disable_subset,
        max_papers=args.max_papers,
    )

    for representation in EXPERIMENT_REPRESENTATIONS:
        _run_representation(
            representation=representation,
            config=config,
            records_path=records_path,
            representation_dir=representation_dir,
            questions_path=questions_path,
            results_dir=results_dir,
            limit=args.limit,
            skip_existing=args.skip_existing,
            force_rebuild=args.force_rebuild,
        )

    summary = aggregate_result_files(
        output_dir=results_dir,
        representation_order=config["evaluation"]["representation_order"],
    )
    print(f"Aggregated {len(summary['rows'])} result rows into {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
