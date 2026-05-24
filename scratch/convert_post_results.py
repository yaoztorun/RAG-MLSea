"""
One-off converter: takes a post_results.json produced by run_post_retrieval_stage.py
and produces a format compatible with evaluate_generation.py.

The key mapping is:
    post_results.json["per_question"][*]["input_candidates"]
    ->
    output["per_question"][*]["results"]

No production code is modified. Run once, then point evaluate_generation.py at the output.
"""
import json
from pathlib import Path


INPUT_PATH  = Path("data/results/post_retrieval/ablation/pure_semantic_dense/post_results.json")
OUTPUT_PATH = Path("data/intermediate/post_retrieval/no_ce_input_for_eval.json")


def main() -> None:
    print(f"Reading: {INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    converted_questions = []
    for entry in data.get("per_question", []):
        # evaluate_generation.py reads entry.get("results", [])
        # map input_candidates (all 10, before filtering) so the eval pipeline
        # can redo filtering + no-CE ranking identically to run_post_retrieval_stage
        converted_questions.append({
            "question_id": entry.get("question_id"),
            "question":    entry.get("question"),
            # "results" is what evaluate_generation.py reads
            "results":     entry.get("input_candidates", []),
        })

    output = {
        "method_name":     data.get("method_name", "pure_semantic_dense"),
        "total_questions": len(converted_questions),
        "per_question":    converted_questions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Written {len(converted_questions)} questions to: {OUTPUT_PATH}")
    print("Next step:")
    print(
        "  python -m src.post_retrieval.scripts.run_evaluate_generation "
        f"--retrieval-results-path {OUTPUT_PATH} "
        "--skip-cross-encoder "
        "--limit 50 "
        "--output-path data/intermediate/post_retrieval/generation_evaluation_no_ce.json"
    )


if __name__ == "__main__":
    main()
