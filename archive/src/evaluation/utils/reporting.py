from typing import Dict


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    print("\nPre-retrieval evaluation results (papers)\n")
    print(f"{'Strategy':<22} {'Hit@1':>8} {'Hit@5':>8} {'MRR':>8} {'NDCG':>8}")
    print("-" * 60)

    for strategy, metrics in results.items():
        print(
            f"{strategy:<22} "
            f"{metrics['hit@1']:.4f} "
            f"{metrics['hit@5']:.4f} "
            f"{metrics['mrr']:.4f} "
            f"{metrics['ndcg']:.4f}"
        )