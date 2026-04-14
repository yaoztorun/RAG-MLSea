# Thesis Evaluation Outputs — README

## Primary Ranking Metric

**NDCG** (Normalized Discounted Cumulative Gain) is the primary ranking metric.
It measures rank-weighted retrieval relevance and penalizes correct answers that
appear lower in the result list. Higher NDCG indicates better overall ranking quality.

## Input Files

All outputs are derived from exactly two authoritative sources:

| File | Description |
|------|-------------|
| `data/retrieval_results/summary.json` | Overall metrics per entity type and representation (16 rows) |
| `data/retrieval_results/summary_by_difficulty.json` | Per-difficulty metrics; rows with `evaluated_questions=0` are excluded |

No existing result files are modified.

## Best Representation per Entity Type (by NDCG)

- **paper**: `enriched_metadata` (NDCG = 0.8225)
- **dataset**: `dataset_title_only` (NDCG = 0.3822)
- **model**: `model_predicate_filtered` (NDCG = 0.8750)

## Tables

### Table 1 — `full_comparison.csv` / `full_comparison.md`
All 16 entity-type × representation combinations with Hit@1, Hit@5, Hit@10, MRR, NDCG.
Best value per metric column is **bolded** in the Markdown version.

### Table 2 — `best_per_entity.csv` / `best_per_entity.md`
One row per entity type showing the representation that achieved the highest NDCG.
Use this for the summary comparison in the thesis.

### Table 3 — `difficulty_breakdown.csv` / `difficulty_breakdown.md`
Per-difficulty (easy / medium / hard) metrics for all representations with evaluated questions.
**Note:** Dataset and model entities have no easy-difficulty questions by evaluation design
(easy questions target paper entities only). Their easy rows are omitted.

## Figures

All figures are saved as both high-resolution PNG (300 dpi) and PDF for direct thesis inclusion.

### Figures 1A–1C — `ndcg_paper`, `ndcg_dataset`, `ndcg_model`
NDCG bar charts, one per entity type, showing performance across that entity's representations.
Representations are kept separate per entity to avoid misleading cross-entity comparisons.

### Figures 2A–2C — `hit1_paper`, `hit1_dataset`, `hit1_model`
Same structure as Figures 1A–1C but for Hit@1.

### Figure 3 — `best_repr_difficulty_breakdown_ndcg`
Grouped bar chart: x-axis = difficulty level (easy / medium / hard),
groups = entity type, y-axis = NDCG.
Only the best representation per entity type is shown.
Missing easy bars for dataset and model are intentional (see Table 3 note above).
