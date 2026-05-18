# Chapter 4 Artifact Manifest

All artifacts cover **pre-retrieval (Section 4.2)** and **retrieval (Section 4.3)** stages only.
No post-retrieval artifacts are included.

---

## Tables

### `tab_evaluation_dataset_summary.tex`
- **Source:** `data/questions/ml_questions_dataset.json + data/results/retrieval/summary_by_difficulty.json`
- **Section:** Section 4.1 (Evaluation Setup)
- **LaTeX label:** `\label{tab:evaluation_dataset_summary}`
- **Suggested caption:** Evaluation dataset summary.
- **Interpretation:** Characterises the 280-question benchmark. Counts for paper, dataset, and model questions are derived from the question_type field. Difficulty distribution comes from the retrieval evaluation (265 answerable questions).

### `tab_best_representation_by_entity.tex`
- **Source:** `data/results/thesis_tables/best_per_entity.csv`
- **Section:** Section 4.2 (Pre-Retrieval Evaluation)
- **LaTeX label:** `\label{tab:best_representation_by_entity}`
- **Suggested caption:** Best entity-centric representation strategy per entity type, evaluated at top-10 candidate entities.
- **Interpretation:** Shows the single best strategy for each entity type. Paper benefits from enriched metadata; models are best served by predicate-filtered text; datasets remain challenging even with the best strategy.

### `tab_pre_retrieval_full_comparison.tex`
- **Source:** `data/results/thesis_tables/full_comparison.csv`
- **Section:** Section 4.2
- **LaTeX label:** `\label{tab:pre_retrieval_full_comparison}`
- **Suggested caption:** Pre-retrieval representation comparison across all 14 strategies and three entity types.
- **Interpretation:** All 14 strategies shown, grouped by entity type with NDCG descending. The spread within each group indicates how sensitive each entity type is to representation choice.

### `tab_pre_retrieval_difficulty.tex`
- **Source:** `data/results/thesis_tables/difficulty_breakdown.csv`
- **Section:** Section 4.2
- **LaTeX label:** `\label{tab:pre_retrieval_difficulty}`
- **Suggested caption:** Difficulty breakdown for the best-performing representation per entity type.
- **Interpretation:** Filters to only the winning strategy per entity type. Reveals that hard questions depress performance significantly regardless of entity type.

### `tab_retrieval_method_comparison.tex`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`
- **Section:** Section 4.3 (Retrieval Evaluation)
- **LaTeX label:** `\label{tab:retrieval_method_comparison}`
- **Suggested caption:** Retrieval method comparison across six methods evaluated on 265 questions over top-10 candidate entities.
- **Interpretation:** RRF+Symbolic achieves the best overall NDCG (+0.0097 over dense). Hybrid methods produce smaller but consistent gains. Hit@1 is highest for Hybrid-Predicate-Aware.

### `tab_retrieval_by_entity_type.tex`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{tab:retrieval_by_entity_type}`
- **Suggested caption:** Retrieval NDCG broken down by entity type.
- **Interpretation:** Dataset entities remain the hardest category; RRF fusion improves dataset NDCG from 0.3822 to 0.4645. Paper and model NDCG are relatively stable across methods.

### `tab_retrieval_by_difficulty.tex`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{tab:retrieval_by_difficulty}`
- **Suggested caption:** Retrieval NDCG broken down by question difficulty.
- **Interpretation:** Hard questions (NDCG ~0.47-0.50) are the clear bottleneck. Easy questions are near ceiling. RRF methods improve medium and unknown categories.

### `tab_retrieval_hit_gap.tex`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{tab:retrieval_hit_gap}`
- **Suggested caption:** Recoverability gap between Hit@1 and Hit@10 across retrieval methods.
- **Interpretation:** RRF-Fusion shows the largest gap (0.1698), meaning it retrieves more correct candidate entities in the top-10 but ranks them lower than the dense baseline does.

### `tab_retrieval_question_type_summary.tex`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{tab:retrieval_question_type_summary}`
- **Suggested caption:** Question-type retrieval summary: selected question types representing best-performing, worst-performing, and RRF-improved categories.
- **Interpretation:** Compact view of 9 representative question types. Full 22-type heatmap is shown in Figure~\ref{fig:retrieval_question_type_heatmap}.

---

## Figures

### `fig_pre_retrieval_ndcg_by_representation.pdf / .png`
- **Source:** `data/results/thesis_tables/full_comparison.csv`
- **Section:** Section 4.2
- **LaTeX label:** `\label{fig:pre_retrieval_ndcg_by_representation}`
- **Suggested caption:** NDCG of all pre-retrieval representation strategies, grouped by entity type. The best strategy per group is highlighted.
- **Interpretation:** 3-panel horizontal bar chart. Consolidates the six per-entity figures previously in data/results/thesis_figures/ into one compact thesis-ready figure. Regenerated for consistent styling.
- **Reuse decision:** Not reused. Existing figures (ndcg_paper/dataset/model) are separate panels; this new figure combines them with consistent style.

### `fig_pre_retrieval_difficulty_ndcg.pdf / .png`
- **Source:** `data/results/thesis_tables/difficulty_breakdown.csv`
- **Section:** Section 4.2
- **LaTeX label:** `\label{fig:pre_retrieval_difficulty_ndcg}`
- **Suggested caption:** NDCG by question difficulty for the best representation per entity type.
- **Interpretation:** Shows how hard questions depress performance across all entity types. Dataset questions are hard at all difficulty levels.
- **Reuse decision:** Not reused. Equivalent to best_repr_difficulty_breakdown_ndcg.png but regenerated with consistent font/color styling.

### `fig_retrieval_method_comparison.pdf / .png`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{fig:retrieval_method_comparison}`
- **Suggested caption:** NDCG and Hit@1 for all six retrieval methods. The dense semantic baseline is shown in grey.
- **Interpretation:** Shows both the ranking quality (NDCG) and top-1 precision (Hit@1) side by side. New artifact — no equivalent in thesis_figures/.
- **Reuse decision:** No existing equivalent.

### `fig_retrieval_by_entity_type.pdf / .png`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{fig:retrieval_by_entity_type}`
- **Suggested caption:** Heatmap of retrieval NDCG by entity type and method.
- **Interpretation:** Reveals that dataset retrieval is weakest and benefits most from RRF fusion. Paper and model NDCG are stable across methods.
- **Reuse decision:** No existing equivalent.

### `fig_retrieval_by_difficulty.pdf / .png`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{fig:retrieval_by_difficulty}`
- **Suggested caption:** Retrieval NDCG by question difficulty across all six methods.
- **Interpretation:** Emphasises that hard questions are the bottleneck. RRF methods marginally improve medium and unknown difficulty while not recovering hard questions significantly.
- **Reuse decision:** No existing equivalent.

### `fig_retrieval_question_type_heatmap.pdf / .png`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{fig:retrieval_question_type_heatmap}`
- **Suggested caption:** Heatmap of NDCG by question type and retrieval method. Question types with fewer than 3 questions are excluded.
- **Interpretation:** Filtered to N>=3 question types (excludes 3 low-count types: model_name_resolution, dataset_to_publication_year, semantic_repository_to_model). Sorted by dense NDCG descending. Reveals which question types benefit from RRF fusion.
- **Reuse decision:** No existing equivalent.

### `fig_retrieval_hit_gap.pdf / .png`
- **Source:** `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv`
- **Section:** Section 4.3
- **LaTeX label:** `\label{fig:retrieval_hit_gap}`
- **Suggested caption:** Hit@1 and Hit@10 for each retrieval method, with the recoverability gap annotated above each method pair.
- **Interpretation:** Supports the argument that RRF fusion sacrifices top-1 precision but recovers more correct candidate entities by rank 10.
- **Reuse decision:** No existing equivalent.

---

## Method Name Mapping

| Short (figure axes) | Display (table rows) | Method ID |
|---|---|---|
| Dense | Dense (baseline) | `pure_semantic_dense` |
| Hybrid-Type | Hybrid: Type Filtering | `hybrid_type_filtering` |
| Hybrid-OneHop | Hybrid: Type + One-Hop | `hybrid_type_onehop_filtering` |
| Hybrid-Pred. | Hybrid: Predicate-Aware | `hybrid_predicate_aware_filtering` |
| RRF-Fusion | RRF: Multi-Repr. Fusion | `optional_rrf_fusion` |
| RRF-Symbolic | RRF: Fusion + Symbolic | `optional_rrf_symbolic` |

## Figure Filtering Rules

- **Figure F (heatmap):** Excludes question types with max count < 3 (`model_name_resolution` N=1, `dataset_to_publication_year` N=2, `semantic_repository_to_model` N=2).
