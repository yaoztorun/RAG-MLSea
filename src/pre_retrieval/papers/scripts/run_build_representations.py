from __future__ import annotations

import argparse
import sys

from src.pre_retrieval.papers.chunking.build_representations import SUPPORTED_REPRESENTATIONS, build_representations
from src.pre_retrieval.shared.config import load_pipeline_config, resolve_records_path, resolve_repo_path
from src.pre_retrieval.shared.gold_target_coverage import load_gold_ids_for_entity_type
from src.pre_retrieval.shared.utils import require_existing_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper text representations from canonical records.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-dir", default="data/intermediate/representations")
    parser.add_argument("--representation", choices=SUPPORTED_REPRESENTATIONS + ["all"], default="title_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--disable-subset", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--questions-path", default="data/questions/ml_questions_dataset.json")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    input_path = resolve_records_path(
        config,
        args.input_path,
        disable_subset=args.disable_subset,
        max_papers=args.max_papers,
    )
    questions_path = resolve_repo_path(args.questions_path)
    try:
        require_existing_input(input_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    gold_target_ids = None
    if questions_path.exists():
        gold_target_ids = load_gold_ids_for_entity_type(questions_path, "paper")
        print(f"[Gold Papers] Loaded {len(gold_target_ids)} gold paper IRIs for force-inclusion.", flush=True)

    if args.representation == "all":
        representation_types = config["evaluation"]["representation_order"]
    else:
        representation_types = [args.representation]

    counts = build_representations(
        records_path=input_path,
        output_dir=resolve_repo_path(args.output_dir),
        representation_types=representation_types,
        representation_config_map=config["representations"],
        limit=args.limit,
        gold_target_ids=gold_target_ids,
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
