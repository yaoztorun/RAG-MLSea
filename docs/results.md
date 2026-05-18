# Chapter 4 Results and Evaluation Plan

> **Scope:** This document is a thesis-ready planning roadmap for Chapter 4 of the thesis
> *Retrieval-Augmented Generation over Machine Learning Knowledge Graphs*.
> All metrics cited are drawn directly from verified repository files. No results are invented.
> Post-retrieval sections are explicitly marked **[TODO — no result files exist]** throughout.

---

## 1. Purpose and Narrative of Chapter 4

Chapter 4 is the empirical core of the thesis. Its purpose is to report, analyse, and interpret
the results of the three-stage KG-RAG pipeline built and described in Chapter 3. The chapter
should answer three things: what the numbers are, what they mean, and why the observed patterns occur.

Chapter 4 must not repeat methodology. It should assume the reader has read Chapter 3 and refer
to it by section label only when a brief pointer is needed. Implementation details should appear
only to the extent that they explain an observed result.

### Connection to Research Questions

**RQ1 — Which entity-centric representation strategy works best for retrieval?**
Section 4.2 answers this directly. The pre-retrieval evaluation covers fourteen representation
strategies across three entity types. The answer is entity-type-dependent: no single strategy
dominates all three entity types, which is itself the primary finding for RQ1.

**RQ2 — Do hybrid symbolic–semantic retrieval methods improve over pure dense retrieval?**
Section 4.3 answers this. Six retrieval methods are compared, ranging from a pure dense baseline
to predicate-aware hybrid methods and multi-representation RRF fusion. The improvements are
real but small in aggregate; the narrative must characterise which sub-populations (entity type,
difficulty, question type) benefit and which do not.

**RQ3 (Post-retrieval) — Does the post-retrieval stage improve answer quality over the
retrieval candidate list alone?**
Section 4.4 addresses this question **if and when post-retrieval results are generated**.
At the time of writing, `data/results/post_retrieval/` does not exist. This section is
structured as a planned section with placeholders.

### Narrative Arc

The chapter should tell this story:
1. Representation quality is the primary determinant of retrieval success; this is established by
   the pre-retrieval ablation (Section 4.2).
2. The best dense baseline — built on those tuned representations — is already strong, which
   limits the headroom for hybrid methods but does not make them irrelevant (Section 4.3).
3. Hard questions and dataset-centric questions are consistent failure modes across all stages,
   pointing to structural sparsity in the knowledge graph (Sections 4.3.6, 4.5).
4. The gap between Hit@1 and Hit@10 indicates that the answer is often in the candidate pool
   even when it is not ranked first, motivating post-retrieval re-ranking (Section 4.5.2).

---

## 2. Final Proposed Table of Contents

```
4. Results and Evaluation

  4.1  Evaluation Setup
       4.1.1  Evaluation Dataset
       4.1.2  Metrics
       4.1.3  Result File Organisation

  4.2  Pre-Retrieval Results: Representation Strategy Evaluation
       4.2.1  Overall Representation Performance
       4.2.2  Paper Representation Results
       4.2.3  Dataset Representation Results
       4.2.4  Model Representation Results
       4.2.5  Performance by Difficulty
       4.2.6  Performance by Question Type
       4.2.7  Summary and Answer to RQ1

  4.3  Retrieval Results: Candidate Ranking Methods
       4.3.1  Overall Retrieval Method Comparison
       4.3.2  Dense Baseline Results
       4.3.3  Hybrid Retrieval Results
       4.3.4  Multi-Representation Fusion Results
       4.3.5  Performance by Entity Type
       4.3.6  Performance by Difficulty
       4.3.7  Performance by Question Type
       4.3.8  Error Analysis of Retrieval Failures
       4.3.9  Summary and Answer to RQ2

  4.4  Post-Retrieval Results  [PLANNED — no result files available]
       4.4.1  Candidate Evidence Availability
       4.4.2  Re-Ranking Results  [TODO]
       4.4.3  Answer Generation Results  [TODO]
       4.4.4  Qualitative Examples  [TODO]
       4.4.5  Summary of Post-Retrieval Findings  [TODO]

  4.5  Cross-Stage Analysis
       4.5.1  How Pre-Retrieval Choices Affect Retrieval
       4.5.2  Recoverability and Top-10 Candidate Quality
       4.5.3  Main Failure Modes

  4.6  Final Summary of Findings
```

---

## 3. Available Result Files and What They Contain

### 3.1 Pre-Retrieval Result Files

| File / Path | Stage | What It Contains | Metrics Available | Entity Types | Recommended Use | Status |
|---|---|---|---|---|---|---|
| `data/results/pre_retrieval_results/paper_results/title_only/results.json` | Pre-retrieval | Per-question metrics for paper `title_only` representation | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Error analysis, qualitative examples | Verified |
| `data/results/pre_retrieval_results/paper_results/abstract_only/results.json` | Pre-retrieval | Per-question metrics for paper `abstract_only` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/paper_results/title_abstract/results.json` | Pre-retrieval | Per-question metrics for paper `title_abstract` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/paper_results/predicate_filtered/results.json` | Pre-retrieval | Per-question metrics for paper `predicate_filtered` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/paper_results/enriched_metadata/results.json` | Pre-retrieval | Per-question metrics for paper `enriched_metadata` (best) | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Primary paper result; error analysis | Verified |
| `data/results/pre_retrieval_results/paper_results/one_hop/results.json` | Pre-retrieval | Per-question metrics for paper `one_hop` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Papers | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/paper_results/*/top10.json` (×6) | Pre-retrieval | Top-10 candidate entity list per question | Ranked candidate IRIs + scores | Papers | Error analysis, qualitative examples | Verified |
| `data/results/pre_retrieval_results/dataset_results/dataset_title_only/results.json` | Pre-retrieval | Per-question metrics for dataset `dataset_title_only` (best) | Hit@1, Hit@5, Hit@10, MRR, NDCG | Datasets | Primary dataset result | Verified |
| `data/results/pre_retrieval_results/dataset_results/dataset_metadata/results.json` | Pre-retrieval | Per-question metrics for dataset `dataset_metadata` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Datasets | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/dataset_results/dataset_predicate_filtered/results.json` | Pre-retrieval | Per-question metrics for dataset `dataset_predicate_filtered` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Datasets | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/dataset_results/dataset_enriched_metadata/results.json` | Pre-retrieval | Per-question metrics for dataset `dataset_enriched_metadata` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Datasets | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/dataset_results/*/top10.json` (×4) | Pre-retrieval | Top-10 candidate entity list per question | Ranked candidate IRIs + scores | Datasets | Error analysis | Verified |
| `data/results/pre_retrieval_results/model_results/model_predicate_filtered/results.json` | Pre-retrieval | Per-question metrics for model `model_predicate_filtered` (best) | Hit@1, Hit@5, Hit@10, MRR, NDCG | Models | Primary model result | Verified |
| `data/results/pre_retrieval_results/model_results/model_metadata/results.json` | Pre-retrieval | Per-question metrics for model `model_metadata` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Models | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/model_results/model_enriched_metadata/results.json` | Pre-retrieval | Per-question metrics for model `model_enriched_metadata` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Models | Ablation comparison | Verified |
| `data/results/pre_retrieval_results/model_results/model_title_only/results.json` | Pre-retrieval | Per-question metrics for model `model_title_only` | Hit@1, Hit@5, Hit@10, MRR, NDCG | Models | Ablation comparison | Verified |

### 3.2 Pre-Retrieval Thesis Table Files

| File / Path | Stage | What It Contains | Metrics Available | Entity Types | Recommended Use | Status |
|---|---|---|---|---|---|---|
| `data/results/thesis_tables/full_comparison.csv` | Pre-retrieval | All 14 entity×representation rows: Hit@1, Hit@5, Hit@10, MRR, NDCG | All five metrics | Papers, Datasets, Models | Table 4.1 — overall pre-retrieval comparison | Verified |
| `data/results/thesis_tables/best_per_entity.csv` | Pre-retrieval | One row per entity type, best-NDCG representation only | All five metrics | Papers, Datasets, Models | Table 4.2 — best representation per entity | Verified |
| `data/results/thesis_tables/difficulty_breakdown.csv` | Pre-retrieval | Per-difficulty (easy/medium/hard) × representation metrics | Hit@1, MRR, NDCG | Papers (all 3 difficulties), Datasets & Models (medium/hard only) | Table 4.5 — performance by difficulty | Verified |
| `data/results/thesis_tables/full_comparison.md` | Pre-retrieval | Markdown version of full_comparison.csv with bolded best values | Same | All | Reference for writing prose | Verified |
| `data/results/thesis_tables/best_per_entity.md` | Pre-retrieval | Markdown version of best_per_entity.csv | Same | All | Reference for writing prose | Verified |
| `data/results/thesis_tables/difficulty_breakdown.md` | Pre-retrieval | Markdown version of difficulty_breakdown.csv | Same | All | Reference for writing prose | Verified |
| `data/results/thesis_tables/README.md` | Meta | Explains table generation, best representations, figure descriptions | N/A | All | Context for writing | Verified |
| `data/results/summary.csv` | Pre-retrieval | Overall aggregate metrics per entity-representation combination | All five metrics | All | Cross-check source | Verified |
| `data/results/summary_by_difficulty.json` | Pre-retrieval | Per-difficulty aggregate metrics | All five metrics | All | Source for difficulty analysis | Verified |
| `data/results/summary_by_category.json` | Pre-retrieval | Per-question-type aggregate metrics | All five metrics | All | Source for question-type analysis | Verified |

### 3.3 Pre-Retrieval Thesis Figures

| File / Path | Stage | What It Shows | Status |
|---|---|---|---|
| `data/results/thesis_figures/ndcg_paper.png` / `.pdf` | Pre-retrieval | NDCG bar chart — all 6 paper representations | Verified — reuse in §4.2.2 |
| `data/results/thesis_figures/ndcg_dataset.png` / `.pdf` | Pre-retrieval | NDCG bar chart — all 4 dataset representations | Verified — reuse in §4.2.3 |
| `data/results/thesis_figures/ndcg_model.png` / `.pdf` | Pre-retrieval | NDCG bar chart — all 4 model representations | Verified — reuse in §4.2.4 |
| `data/results/thesis_figures/hit1_paper.png` / `.pdf` | Pre-retrieval | Hit@1 bar chart — all 6 paper representations | Verified — reuse in §4.2.2 |
| `data/results/thesis_figures/hit1_dataset.png` / `.pdf` | Pre-retrieval | Hit@1 bar chart — all 4 dataset representations | Verified — reuse in §4.2.3 |
| `data/results/thesis_figures/hit1_model.png` / `.pdf` | Pre-retrieval | Hit@1 bar chart — all 4 model representations | Verified — reuse in §4.2.4 |
| `data/results/thesis_figures/best_repr_difficulty_breakdown_ndcg.png` / `.pdf` | Pre-retrieval | Grouped bar: NDCG by difficulty, best representation per entity | Verified — reuse in §4.2.5 |

### 3.4 Retrieval Result Files

| File / Path | Stage | What It Contains | Metrics Available | Entity Types | Recommended Use | Status |
|---|---|---|---|---|---|---|
| `data/results/retrieval/pure_semantic_dense/metrics.json` | Retrieval | Aggregate metrics for dense baseline | Hit@1, Hit@5, Hit@10, MRR, NDCG + breakdowns | All | §4.3.2 dense baseline | Verified |
| `data/results/retrieval/pure_semantic_dense/results.json` | Retrieval | Per-question results for dense baseline | Per-question ranks + metrics | All | Error analysis §4.3.8 | Verified |
| `data/results/retrieval/hybrid_type_filtering/metrics.json` | Retrieval | Aggregate metrics for type-filter hybrid | All five | All | §4.3.3 sanity check | Verified |
| `data/results/retrieval/hybrid_type_onehop_filtering/metrics.json` | Retrieval | Aggregate metrics for one-hop hybrid | All five | All | §4.3.3 | Verified |
| `data/results/retrieval/hybrid_predicate_aware_filtering/metrics.json` | Retrieval | Aggregate metrics for predicate-aware hybrid | All five | All | §4.3.3 | Verified |
| `data/results/retrieval/optional_rrf_fusion/metrics.json` | Retrieval | Aggregate metrics for RRF fusion | All five | All | §4.3.4 | Verified |
| `data/results/retrieval/optional_rrf_symbolic/metrics.json` | Retrieval | Aggregate metrics for RRF + symbolic | All five | All | §4.3.4 | Verified |
| `data/results/retrieval/*/results.json` (×7) | Retrieval | Per-question ranked candidates + scores | Per-question | All | Error analysis | Verified |
| `data/results/retrieval/type_filtering/` | Retrieval | Internal type-filter control (not a thesis method) | All five | All | Background only; not for a results table | Verified |

### 3.5 Retrieval Thesis Table Files

| File / Path | Stage | What It Contains | Recommended Use | Status |
|---|---|---|---|---|
| `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` | Retrieval | All 6 methods × 5 metrics + delta_NDCG + interpretation label | Table 4.6 — main retrieval comparison | Verified |
| `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv` | Retrieval | Method × entity type NDCG matrix (3 entity types) | Table 4.7 — retrieval NDCG by entity type | Verified |
| `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` | Retrieval | Method × difficulty NDCG matrix (easy/medium/hard/unknown) | Table 4.8 — retrieval NDCG by difficulty | Verified |
| `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` | Retrieval | Method × question type: NDCG, Hit@1, Hit@10 (22 question types) | Table 4.9 — retrieval NDCG by question type | Verified |
| `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` | Retrieval | Method × Hit@1/Hit@10/gap — precision–recall tradeoff | Table 4.10 — Hit@1 vs Hit@10 gap | Verified |
| `data/results/retrieval/thesis_tables/retrieval_interpretation.md` | Meta | Framing guidance for retrieval results narrative | Writing reference | Verified |
| `data/results/retrieval/summary.csv` | Retrieval | Top-level aggregate: method, n, Hit@1, Hit@5, Hit@10, MRR, NDCG | Cross-check source | Verified |
| `data/results/retrieval/summary_by_entity_type.json` | Retrieval | Per-entity-type breakdown for all methods | Source for §4.3.5 | Verified |
| `data/results/retrieval/summary_by_difficulty.json` | Retrieval | Per-difficulty breakdown for all methods | Source for §4.3.6 | Verified |
| `data/results/retrieval/summary_by_question_type.json` | Retrieval | Per-question-type breakdown for all methods | Source for §4.3.7 | Verified |

### 3.6 Post-Retrieval Result Files

| File / Path | Stage | What It Contains | Status |
|---|---|---|---|
| `data/results/post_retrieval/` | Post-retrieval | Expected output directory | **Missing — directory does not exist** |
| Generation result JSON | Post-retrieval | Per-question: generated answer, SAS, ROUGE-L, LLM-judge score | **TODO — not generated** |
| Re-ranking result JSON | Post-retrieval | Per-question: re-ranked candidate list from cross-encoder | **TODO — not generated** |

---

## 4. Tables to Include in Chapter 4

### Table 4.1 — Full Pre-Retrieval Representation Comparison

| Field | Detail |
|---|---|
| **Table number** | 4.1 |
| **Proposed title** | Retrieval Performance of All Entity-Centric Representations |
| **Section** | 4.2.1 — Overall Representation Performance |
| **Source file** | `data/results/thesis_tables/full_comparison.csv` |
| **Columns** | Entity Type, Representation, Hit@1, Hit@5, Hit@10, MRR, NDCG |
| **Rows** | 14 rows (6 paper + 4 dataset + 4 model strategies) |
| **Purpose** | Provide a complete ablation table; let the reader see all strategies at once before focused per-entity discussion |
| **Caption** | *Retrieval performance of all entity-centric representation strategies evaluated on 265 answerable questions. NDCG is the primary ranking metric. Bold values indicate the best result within each entity type. Dataset and model strategies are evaluated on their respective question subsets; paper strategies on paper questions.* |
| **Notes** | Bold best NDCG per entity group. Do not compare paper vs. dataset metrics directly — question populations differ. |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.2 — Best Representation per Entity Type

| Field | Detail |
|---|---|
| **Table number** | 4.2 |
| **Proposed title** | Best Representation Strategy per Entity Type |
| **Section** | 4.2.1 — Overall Representation Performance |
| **Source file** | `data/results/thesis_tables/best_per_entity.csv` |
| **Columns** | Entity Type, Best Representation, Hit@1, Hit@5, Hit@10, MRR, NDCG |
| **Rows** | 3 rows (paper, dataset, model) |
| **Purpose** | Provide a concise summary of the pre-retrieval winner per entity; directly answers RQ1 |
| **Caption** | *Best-performing representation strategy per entity type, selected by highest NDCG. These representations are used as inputs to the retrieval stage.* |
| **Notes** | Values: paper `enriched_metadata` NDCG=0.8225; dataset `dataset_title_only` NDCG=0.3822; model `model_predicate_filtered` NDCG=0.8750. |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.3 — Paper Representation Results

| Field | Detail |
|---|---|
| **Table number** | 4.3 |
| **Proposed title** | Paper Representation Evaluation: All Six Strategies |
| **Section** | 4.2.2 — Paper Representation Results |
| **Source file** | `data/results/thesis_tables/full_comparison.csv` (paper rows only) |
| **Columns** | Representation, Hit@1, Hit@5, Hit@10, MRR, NDCG |
| **Rows** | 6 rows |
| **Purpose** | Focused comparison for paper entity type; support the narrative in §4.2.2 |
| **Caption** | *Retrieval performance of six paper representation strategies on 178 paper-centric questions (265 answerable questions include paper, dataset, and model; paper subset size verified in results.json files).* |
| **Notes** | Highlight that `abstract_only` performs worst (NDCG=0.5438) and `enriched_metadata` best (0.8225). Note `title_only` achieves strong Hit@1 (0.7022) due to exact-match bias. |
| **Exists?** | Subset of existing CSV; LaTeX table must be generated |

### Table 4.4 — Dataset Representation Results

| Field | Detail |
|---|---|
| **Table number** | 4.4 |
| **Proposed title** | Dataset Representation Evaluation: All Four Strategies |
| **Section** | 4.2.3 — Dataset Representation Results |
| **Source file** | `data/results/thesis_tables/full_comparison.csv` (dataset rows only) |
| **Columns** | Representation, Hit@1, Hit@5, Hit@10, MRR, NDCG |
| **Rows** | 4 rows |
| **Purpose** | Show that `dataset_title_only` outperforms enriched strategies — a counter-intuitive finding explained by sparse metadata |
| **Caption** | *Retrieval performance of four dataset representation strategies. Dataset entities have sparse metadata in MLSea; results reflect that sparsity.* |
| **Notes** | Key values: `dataset_title_only` NDCG=0.3822 vs `dataset_enriched_metadata` NDCG=0.3243. Enriched metadata underperforms — must explain in prose. |
| **Exists?** | Subset of existing CSV; LaTeX table must be generated |

### Table 4.5 — Model Representation Results

| Field | Detail |
|---|---|
| **Table number** | 4.5 |
| **Proposed title** | Model Representation Evaluation: All Four Strategies |
| **Section** | 4.2.4 — Model Representation Results |
| **Source file** | `data/results/thesis_tables/full_comparison.csv` (model rows only) |
| **Columns** | Representation, Hit@1, Hit@5, Hit@10, MRR, NDCG |
| **Rows** | 4 rows |
| **Purpose** | Show strong `model_predicate_filtered` performance; show why graph-structural predicates matter for models |
| **Caption** | *Retrieval performance of four model representation strategies. Model entities rely heavily on structured repository links and predicate context.* |
| **Notes** | Key values: `model_predicate_filtered` NDCG=0.8750, `model_title_only` NDCG=0.4465. Large gap shows title signal alone is insufficient for model identification. |
| **Exists?** | Subset of existing CSV; LaTeX table must be generated |

### Table 4.6 — Performance by Difficulty (Pre-Retrieval)

| Field | Detail |
|---|---|
| **Table number** | 4.6 |
| **Proposed title** | Representation Performance by Question Difficulty (Best Strategy per Entity) |
| **Section** | 4.2.5 — Performance by Difficulty |
| **Source file** | `data/results/thesis_tables/difficulty_breakdown.csv` (best-strategy rows only) |
| **Columns** | Entity Type, Difficulty, Representation, Hit@1, MRR, NDCG |
| **Rows** | ~7–9 rows (paper: easy/medium/hard; dataset: medium/hard; model: medium/hard) |
| **Purpose** | Show that hard questions are dramatically harder — especially for papers — and that dataset easy questions are absent (by evaluation design) |
| **Caption** | *Retrieval performance segmented by question difficulty for the best representation per entity type. Easy rows are not available for dataset and model entities, as easy-difficulty questions exclusively target paper entities in this evaluation.* |
| **Notes** | Key finding: paper `enriched_metadata` easy=0.9815, medium=0.9732, hard=0.4931 NDCG. Hard questions drop sharply. |
| **Exists?** | CSV exists; need to filter to best-strategy rows; LaTeX table must be generated |

### Table 4.7 — Retrieval Method Comparison (Main)

| Field | Detail |
|---|---|
| **Table number** | 4.7 |
| **Proposed title** | Retrieval Method Comparison: Overall Performance on 265 Questions |
| **Section** | 4.3.1 — Overall Retrieval Method Comparison |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` |
| **Columns** | Method, Group, Hit@1, Hit@5, Hit@10, MRR, NDCG, ΔNDCG vs Dense |
| **Rows** | 6 rows (5 thesis methods + note about type_filtering as internal control) |
| **Purpose** | Primary retrieval comparison table; establishes that improvements over dense baseline are modest |
| **Caption** | *Retrieval performance of six candidate generation methods on 265 answerable questions. $\Delta$NDCG is measured relative to the pure semantic dense baseline. Methods are grouped by strategy family.* |
| **Notes** | Do not include `type_filtering` as a separate row — it is an internal control. `hybrid_type_filtering == pure_semantic_dense` because collections are entity-type-pure; this is the expected sanity check result, not a failure. |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.8 — Retrieval Performance by Entity Type

| Field | Detail |
|---|---|
| **Table number** | 4.8 |
| **Proposed title** | Retrieval NDCG by Entity Type and Method |
| **Section** | 4.3.5 — Performance by Entity Type |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv` |
| **Columns** | Method, Paper NDCG, Dataset NDCG, Model NDCG |
| **Rows** | 6 rows |
| **Purpose** | Show that RRF fusion helps datasets (+0.08 over dense) but hurts models slightly |
| **Caption** | *NDCG segmented by entity type for each retrieval method. Dataset entities benefit most from multi-representation fusion; model entities show a modest decline under RRF.* |
| **Notes** | Key values (verified): dataset dense=0.382, RRF symbolic=0.465; model dense=0.875, RRF=0.850. |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.9 — Retrieval Performance by Difficulty

| Field | Detail |
|---|---|
| **Table number** | 4.9 |
| **Proposed title** | Retrieval NDCG by Question Difficulty and Method |
| **Section** | 4.3.6 — Performance by Difficulty |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` |
| **Columns** | Method, Easy NDCG, Medium NDCG, Hard NDCG, Unknown NDCG |
| **Rows** | 6 rows |
| **Purpose** | Show that all methods plateau near-perfectly on easy questions; hard questions are the bottleneck |
| **Caption** | *Retrieval NDCG segmented by question difficulty. All methods achieve near-ceiling performance on easy questions; hard questions expose the limits of current representations and ranking strategies.* |
| **Notes** | Key values: easy ≈ 0.981–0.991 (all methods), hard ≈ 0.467–0.497. Marginal hard-question gains from hybrid methods. |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.10 — Retrieval Performance by Question Type

| Field | Detail |
|---|---|
| **Table number** | 4.10 |
| **Proposed title** | Retrieval Performance by Question Type: NDCG, Hit@1, Hit@10 for Selected Methods |
| **Section** | 4.3.7 — Performance by Question Type |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` |
| **Columns** | Question Type, N, Method 1 NDCG, Method 2 NDCG, … (select 2–3 methods for readability) |
| **Rows** | 22 question types |
| **Purpose** | Show which question types are trivially solved (paper_to_tasks, paper_to_author_count) and which consistently fail (tasks_to_dataset, paper_by_author_and_task) |
| **Caption** | *Retrieval performance by question type for selected methods. Only question types with $n \geq 2$ are shown. Performance varies widely across question types, reflecting differences in the distinctiveness of the target entity's representation text.* |
| **Notes** | Consider showing only `pure_semantic_dense` and `optional_rrf_symbolic` columns to keep the table legible. Flag question types with NDCG=0 for the dense method. Use `sidewaystable` or reduce font size. |
| **Exists?** | CSV exists with all methods; LaTeX table must be generated; may need to be split or abbreviated |

### Table 4.11 — Hit@1 vs Hit@10 Gap (Precision–Recall Tradeoff)

| Field | Detail |
|---|---|
| **Table number** | 4.11 |
| **Proposed title** | Hit@1 and Hit@10 Gap by Retrieval Method |
| **Section** | 4.5.2 — Recoverability and Top-10 Candidate Quality |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` |
| **Columns** | Method, Hit@1, Hit@10, Gap (Hit@10 − Hit@1) |
| **Rows** | 6 rows |
| **Purpose** | Show that the answer is frequently present in the top-10 but not at rank 1; motivates post-retrieval re-ranking |
| **Caption** | *Hit@1 versus Hit@10 for each retrieval method. The gap represents the fraction of questions where the gold entity is retrieved within the top 10 but not at rank 1, indicating that re-ranking has room to improve final answer accuracy.* |
| **Notes** | Key values: dense gap=0.109; RRF fusion gap=0.170 (RRF broadens recall at cost of precision). |
| **Exists?** | CSV exists; LaTeX table must be generated |

### Table 4.12 — Error Analysis Summary

| Field | Detail |
|---|---|
| **Table number** | 4.12 |
| **Proposed title** | Retrieval Failure Analysis: Category, Frequency, Representative Examples |
| **Section** | 4.3.8 — Error Analysis of Retrieval Failures |
| **Source file** | `data/results/retrieval/pure_semantic_dense/results.json` (and other method results.json files) |
| **Columns** | Failure Category, Entity Type, Frequency (N), Example Question Type, Root Cause |
| **Rows** | ~5–8 rows covering distinct failure categories |
| **Purpose** | Categorise and quantify failure modes to support the limitations narrative |
| **Caption** | *Categorisation of retrieval failures for the pure semantic dense baseline. Failures are drawn from questions where the gold entity is absent from the top-10 candidate list.* |
| **Notes** | Frequency counts must be computed from results.json files — do not invent numbers. Failure categories: gold absent from top-10; gold present but ranked 6–10; semantically similar distractor at rank 1; question type with zero retrievable signal. |
| **Exists?** | Must be generated from per-question results.json files |

### Table 4.13 — Post-Retrieval Results [TODO]

| Field | Detail |
|---|---|
| **Table number** | 4.13 |
| **Proposed title** | Post-Retrieval Answer Generation Performance |
| **Section** | 4.4.3 — Answer Generation Results |
| **Source file** | `data/results/post_retrieval/` — **does not exist** |
| **Columns** | Method, SAS, ROUGE-L, LLM-Judge Correct (%) |
| **Status** | **TODO — result file missing. Do not include in thesis until generated.** |

---

## 5. Figures to Include in Chapter 4

### Figure 4.1 — NDCG by Paper Representation

| Field | Detail |
|---|---|
| **Figure number** | 4.1 |
| **Proposed title** | NDCG Across Paper Representation Strategies |
| **Section** | 4.2.2 — Paper Representation Results |
| **Source file** | `data/results/thesis_figures/ndcg_paper.pdf` / `.png` |
| **Plot type** | Horizontal bar chart |
| **x-axis** | NDCG (0 to 1) |
| **y-axis** | Representation strategy (6 bars) |
| **Shows** | Relative ranking quality across paper representations; `enriched_metadata` clearly best |
| **Why useful** | Immediately visual; supports the prose before the reader looks at numbers |
| **Caption** | *NDCG at rank 10 for six paper representation strategies. \texttt{enriched\_metadata} achieves the highest ranking quality (NDCG = 0.8225); \texttt{abstract\_only} performs worst (NDCG = 0.5438), confirming that title signal is essential for paper retrieval.* |
| **Exists?** | **Yes — reuse `data/results/thesis_figures/ndcg_paper.pdf`** |

### Figure 4.2 — Hit@1 by Paper Representation

| Field | Detail |
|---|---|
| **Figure number** | 4.2 |
| **Proposed title** | Hit@1 Across Paper Representation Strategies |
| **Section** | 4.2.2 — Paper Representation Results |
| **Source file** | `data/results/thesis_figures/hit1_paper.pdf` |
| **Plot type** | Horizontal bar chart |
| **Shows** | Precision at rank 1 per paper strategy |
| **Caption** | *Hit@1 for six paper representation strategies. \texttt{enriched\_metadata} achieves the highest precision at rank 1 (Hit@1 = 0.775).* |
| **Exists?** | **Yes — reuse `data/results/thesis_figures/hit1_paper.pdf`** |

### Figure 4.3 — NDCG by Dataset Representation

| Field | Detail |
|---|---|
| **Figure number** | 4.3 |
| **Proposed title** | NDCG Across Dataset Representation Strategies |
| **Section** | 4.2.3 — Dataset Representation Results |
| **Source file** | `data/results/thesis_figures/ndcg_dataset.pdf` |
| **Plot type** | Horizontal bar chart |
| **Shows** | Counter-intuitive result: `dataset_title_only` outperforms enriched and predicate strategies |
| **Caption** | *NDCG for four dataset representation strategies. \texttt{dataset\_title\_only} achieves the highest NDCG (0.3822), suggesting that additional metadata introduces noise rather than signal for sparse dataset entities.* |
| **Exists?** | **Yes — reuse `data/results/thesis_figures/ndcg_dataset.pdf`** |

### Figure 4.4 — NDCG by Model Representation

| Field | Detail |
|---|---|
| **Figure number** | 4.4 |
| **Proposed title** | NDCG Across Model Representation Strategies |
| **Section** | 4.2.4 — Model Representation Results |
| **Source file** | `data/results/thesis_figures/ndcg_model.pdf` |
| **Plot type** | Horizontal bar chart |
| **Shows** | Large gap between `model_predicate_filtered` (NDCG=0.875) and title-only (0.446) |
| **Caption** | *NDCG for four model representation strategies. \texttt{model\_predicate\_filtered} substantially outperforms simpler strategies, confirming the importance of graph-structural predicate context for model entity retrieval.* |
| **Exists?** | **Yes — reuse `data/results/thesis_figures/ndcg_model.pdf`** |

### Figure 4.5 — NDCG by Difficulty, Best Representation per Entity

| Field | Detail |
|---|---|
| **Figure number** | 4.5 |
| **Proposed title** | NDCG by Question Difficulty for Best Representation per Entity Type |
| **Section** | 4.2.5 — Performance by Difficulty |
| **Source file** | `data/results/thesis_figures/best_repr_difficulty_breakdown_ndcg.pdf` |
| **Plot type** | Grouped bar chart (groups = entity type, x-axis = difficulty) |
| **Shows** | Near-ceiling performance on easy/medium; hard questions drop sharply for papers |
| **Caption** | *NDCG by question difficulty for the best-performing representation of each entity type. Easy bars for dataset and model entities are absent because all easy-difficulty questions target paper entities. Hard questions represent the primary performance bottleneck.* |
| **Exists?** | **Yes — reuse `data/results/thesis_figures/best_repr_difficulty_breakdown_ndcg.pdf`** |

### Figure 4.6 — Retrieval Method Comparison Bar Chart [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.6 |
| **Proposed title** | NDCG and Hit@1 Comparison Across Retrieval Methods |
| **Section** | 4.3.1 — Overall Retrieval Method Comparison |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` |
| **Plot type** | Grouped bar chart (NDCG + Hit@1, one group per method) or side-by-side bars |
| **x-axis** | Retrieval method (6 methods) |
| **y-axis** | Metric value (0 to 1) |
| **Shows** | Small but meaningful NDCG differences across methods; Hit@1 vs NDCG trade-off for RRF |
| **Why useful** | Makes the comparison immediately visible; the table alone is easy to misread |
| **Caption** | *NDCG and Hit@1 for six retrieval methods on 265 answerable questions. The pure semantic dense baseline serves as the reference. Marginal NDCG improvements from hybrid methods mask larger Hit@1 vs Hit@10 trade-offs visible in Figure~\ref{fig:retrieval_hit_gap}.* |
| **Exists?** | **No — must generate from retrieval_main_comparison.csv** |

### Figure 4.7 — Retrieval NDCG by Entity Type [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.7 |
| **Proposed title** | Retrieval NDCG by Entity Type for Each Method |
| **Section** | 4.3.5 — Performance by Entity Type |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv` |
| **Plot type** | Grouped bar chart (groups = entity type, bars = methods) or heatmap |
| **x-axis** | Retrieval method |
| **y-axis** | NDCG |
| **Shows** | RRF fusion helps datasets; models slightly hurt by RRF |
| **Caption** | *Retrieval NDCG segmented by entity type. Dataset entities benefit from multi-representation fusion; model entities are best served by the dense baseline due to the specificity of their predicate-filtered representations.* |
| **Exists?** | **No — must generate** |

### Figure 4.8 — Retrieval NDCG by Difficulty [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.8 |
| **Proposed title** | Retrieval NDCG by Question Difficulty |
| **Section** | 4.3.6 — Performance by Difficulty |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` |
| **Plot type** | Grouped bar chart (groups = difficulty, bars = methods) |
| **x-axis** | Difficulty (easy, medium, hard) |
| **y-axis** | NDCG |
| **Shows** | All methods converge at ceiling on easy; diverge on medium; bottom out on hard |
| **Caption** | *Retrieval NDCG by question difficulty. Hard questions consistently yield NDCG below 0.50 across all retrieval methods, indicating that the bottleneck lies in representation quality rather than ranking strategy.* |
| **Exists?** | **No — must generate** |

### Figure 4.9 — Question Type Performance Heatmap [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.9 |
| **Proposed title** | Retrieval NDCG Heatmap: Question Type vs Method |
| **Section** | 4.3.7 — Performance by Question Type |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` |
| **Plot type** | Heatmap (rows = question types, columns = methods, cells = NDCG) |
| **x-axis** | Retrieval method (6 columns) |
| **y-axis** | Question type (22 rows) |
| **Shows** | Which question types are systematically unresolvable and which are trivially solved |
| **Why useful** | Reveals structural failure modes invisible in aggregate metrics |
| **Caption** | *NDCG heatmap across 22 question types and six retrieval methods. Dark cells indicate zero or near-zero NDCG; brighter cells indicate strong performance. Question types \texttt{tasks\_to\_dataset} and \texttt{semantic\_task\_to\_dataset} show near-zero NDCG for the dense baseline but modest gains under RRF fusion.* |
| **Exists?** | **No — must generate** |

### Figure 4.10 — Hit@1 vs Hit@10 Gap [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.10 |
| **Proposed title** | Hit@1 and Hit@10 by Retrieval Method with Recoverability Gap |
| **Section** | 4.5.2 — Recoverability and Top-10 Candidate Quality |
| **Source file** | `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` |
| **Plot type** | Stacked or overlaid bar chart showing Hit@1 (solid) and Hit@10 (lighter); gap highlighted |
| **x-axis** | Method |
| **y-axis** | Hit rate (0 to 1) |
| **Shows** | RRF methods widen the gap (more recall, less precision); predicate-aware hybrid narrows the gap |
| **Caption** | *Hit@1 and Hit@10 for each retrieval method. The shaded region between Hit@1 and Hit@10 represents candidate lists where the gold entity is present but not top-ranked. This gap quantifies the headroom available to a downstream re-ranker.* |
| **Exists?** | **No — must generate** |

### Figure 4.11 — Failure Mode Distribution [MUST GENERATE]

| Field | Detail |
|---|---|
| **Figure number** | 4.11 |
| **Proposed title** | Distribution of Retrieval Failure Categories |
| **Section** | 4.3.8 — Error Analysis |
| **Source file** | `data/results/retrieval/pure_semantic_dense/results.json` (computed from file) |
| **Plot type** | Horizontal stacked bar or pie chart showing failure category breakdown |
| **Shows** | Proportion of questions failing because: gold absent from top-10 vs. gold present but ranked low vs. other |
| **Caption** | *Distribution of retrieval failure categories for the pure semantic dense baseline. Failures are defined as questions where Hit@10 = 0 or where the gold entity is present in the top 10 but ranked below position 1.* |
| **Exists?** | **No — must generate from per-question JSON** |

---

## 6. Detailed Section-by-Section Writing Plan

### 4.1 Evaluation Setup

#### 4.1.1 Evaluation Dataset
- **Goal:** Establish the scale and composition of the evaluation set
- **Results to present:** 280 total questions; 265 answerable used for quantitative metrics; 15 unanswerable excluded. Distribution: 178 paper questions, 57 dataset questions, 30 model questions. 35 question types. Difficulty distribution: Easy (40), Medium (103), Hard (77), unlabelled (45).
- **Tables/Figures:** None — prose only
- **Key interpretation:** State why unanswerable questions are excluded. Note that difficulty labels are a property of the question set, not a result.
- **What not to include:** Do not explain how questions were generated; that belongs in Chapter 3.
- **Source files:** `data/questions/ml_questions_dataset.json` (for verification only; do not re-derive in prose)

#### 4.1.2 Metrics
- **Goal:** Define metrics concisely for a reader who has not memorised Chapter 3
- **Results to present:** Hit@k for k ∈ {1, 5, 10}; MRR; NDCG@10 (primary). Formulae are in Chapter 3 §3.3 — reference by label, do not repeat.
- **Key point:** Single gold target per question. NDCG at rank 10 is primary because it penalises low-ranked correct answers.
- **What not to include:** Do not re-derive formulas. Do not discuss metric tradeoffs at length.
- **Source files:** Chapter 3 §3.3 (reference only)

#### 4.1.3 Result File Organisation
- **Goal:** Orient the reader so they could reproduce any table or figure from source files
- **Results to present:** Three-stage result directory structure. Pre-retrieval under `data/results/pre_retrieval_results/`; retrieval under `data/results/retrieval/`; post-retrieval planned under `data/results/post_retrieval/` (not yet available).
- **What not to include:** Internal implementation details. Do not list all 36 JSON files.

---

### 4.2 Pre-Retrieval Results

#### 4.2.1 Overall Representation Performance
- **Goal:** Present the complete ablation across all 14 strategies; introduce the entity-dependence finding
- **Results:** Table 4.1 (full comparison); Table 4.2 (best per entity)
- **Key interpretation:** No single strategy dominates across entity types. This is the primary finding for RQ1. NDCG ranges from 0.192 (`dataset_predicate_filtered`) to 0.875 (`model_predicate_filtered`). The entity type accounts for more variance than the strategy choice within a type.
- **What not to include:** Do not explain the implementation of each representation; that is Chapter 3 §4.4.

#### 4.2.2 Paper Representation Results
- **Goal:** Detailed analysis for the paper entity type (largest subset, most strategies)
- **Results:** Table 4.3; Figure 4.1 (NDCG); Figure 4.2 (Hit@1)
- **Key interpretation points:**
  - `enriched_metadata` is best overall (NDCG=0.8225, Hit@1=0.7753)
  - `abstract_only` is worst (NDCG=0.5438) — removing the title signal degrades retrieval severely; titles are the primary matching signal for paper questions
  - `predicate_filtered` (NDCG=0.7745) is competitive despite being structurally simpler than `enriched_metadata`; this suggests RDF predicate information contributes to disambiguation
  - `title_only` achieves Hit@1=0.7022 — strong but below `enriched_metadata`, showing that metadata adds rank-1 precision
  - `one_hop` (NDCG=0.7642) improves recall (Hit@10=0.820) but not precision (Hit@1=0.714) over title-only; graph context helps with broader recall but not top-rank precision
- **What not to include:** Do not explain how representations are constructed.
- **Source files:** `data/results/thesis_tables/full_comparison.csv`; `data/results/thesis_figures/ndcg_paper.pdf`

#### 4.2.3 Dataset Representation Results
- **Goal:** Explain the counter-intuitive finding that a simpler representation outperforms enriched ones
- **Results:** Table 4.4; Figure 4.3
- **Key interpretation points:**
  - `dataset_title_only` is best (NDCG=0.3822, Hit@1=0.2807)
  - `dataset_enriched_metadata` underperforms title-only (NDCG=0.3243); enriching with sparse or noisy metadata degrades embedding quality
  - `dataset_predicate_filtered` is worst (NDCG=0.1919); aggressive filtering may remove the title — the only reliable anchor
  - All dataset strategies show substantially lower performance than paper and model strategies — a structural finding driven by sparse dataset metadata in MLSea
  - Dataset questions require recall of a specific dataset name, which the embedding model can match well from a clean title but not from a noisy enriched text
- **What not to include:** Do not repeat the metadata sparsity explanation from Chapter 3; summarise its consequence here.
- **Source files:** `data/results/thesis_tables/full_comparison.csv`; `data/results/thesis_figures/ndcg_dataset.pdf`

#### 4.2.4 Model Representation Results
- **Goal:** Explain the dramatic gap between predicate-filtered and other model strategies
- **Results:** Table 4.5; Figure 4.4
- **Key interpretation points:**
  - `model_predicate_filtered` achieves NDCG=0.8750 — the highest of any entity-type/strategy combination in the pre-retrieval stage
  - `model_title_only` yields NDCG=0.4465 — less than half — showing model names alone are insufficient for retrieval
  - `model_enriched_metadata` (NDCG=0.6916) is intermediate; enrichment helps but unfiltered noise hurts
  - Predicate filtering for models includes repository links and linked entity context, which are the primary signals in model-type questions (e.g., `repository_to_model`)
  - Hard model questions achieve NDCG=0.943 with `model_predicate_filtered` — the highest hard-question performance across all entities
- **Source files:** `data/results/thesis_tables/full_comparison.csv`; `data/results/thesis_figures/ndcg_model.pdf`; `data/results/thesis_tables/difficulty_breakdown.csv`

#### 4.2.5 Performance by Difficulty
- **Goal:** Establish difficulty as a major performance axis; connect to failure mode analysis
- **Results:** Table 4.6; Figure 4.5 (existing)
- **Key interpretation points:**
  - Easy paper questions: near-ceiling across most strategies (enriched_metadata NDCG=0.9815, one_hop NDCG=0.9908)
  - Medium paper questions: slight drop (enriched_metadata NDCG=0.9732)
  - Hard paper questions: sharp drop (enriched_metadata NDCG=0.4931, abstract_only NDCG=0.2733)
  - `enriched_metadata` is most robust for hard paper questions; `title_only` collapses (NDCG=0.1046 on hard)
  - Dataset hard questions: all strategies drop; `dataset_enriched_metadata` (NDCG=0.3046) narrowly best on hard despite being worst overall
  - Model hard questions: `model_predicate_filtered` remains strong (NDCG=0.943); models may benefit from deterministic predicate signals that are difficulty-independent
  - Note: easy rows absent for datasets and models (by evaluation design — verified from results.json)
- **Source files:** `data/results/thesis_tables/difficulty_breakdown.csv`

#### 4.2.6 Performance by Question Type
- **Goal:** Identify question types where pre-retrieval representation succeeds or fails categorically
- **Results:** Reference Table 4.10 (retrieval by question type) — also valid for pre-retrieval because the underlying pre-retrieval representations feed the dense baseline
- **Key interpretation points:**
  - `paper_to_tasks`, `paper_to_authors`, `paper_to_implementation`, `paper_to_author_count`, `paper_to_keywords`: near-perfect (NDCG ≥ 0.95 for dense baseline fed by best representations)
  - `paper_by_author_and_task`: NDCG=0.133 — highly ambiguous multi-predicate questions; semantically similar papers cluster in embedding space
  - `tasks_to_dataset`: NDCG=0.0 for dense — the question describes a set of tasks and requires the dataset associated with all of them; not solvable by vector similarity to title
  - `semantic_task_to_dataset`: NDCG=0.071 for dense — same structural failure
  - `dataset_to_task_membership`: NDCG=0.246 — low because the question tests membership, not similarity
- **What not to include:** Do not list all 22 question types exhaustively in prose; refer to Table 4.10 for the full breakdown.
- **Source files:** `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` (dense baseline rows)

#### 4.2.7 Summary and Answer to RQ1
- **Goal:** State the direct answer to RQ1 clearly
- **Text guidance:** The answer is: *entity-type determines the optimal representation strategy; there is no universal best strategy*. Papers benefit from rich structured metadata (`enriched_metadata`). Datasets benefit from minimal, clean title-only representations because their metadata is too sparse to embed reliably. Models benefit from predicate-filtered graph context that includes repository links.
- **What not to include:** Do not introduce new evidence here.

---

### 4.3 Retrieval Results

#### 4.3.1 Overall Retrieval Method Comparison
- **Goal:** Present the aggregate comparison; establish the scale of improvement over the baseline
- **Results:** Table 4.7; Figure 4.6 (must generate)
- **Key interpretation points:**
  - The dense baseline (NDCG=0.7337) is already strong because it uses the best pre-retrieval representations from §4.2
  - `hybrid_type_filtering` produces identical results (NDCG=0.7337); this is expected and serves as a sanity check — the collections are already entity-type-pure
  - `hybrid_predicate_aware_filtering` gives the best Hit@1 among hybrid methods (0.6868 vs 0.6717)
  - `optional_rrf_symbolic` gives the best overall NDCG (0.7434, +0.0097)
  - Improvements are real but small in aggregate; interpret as: the dense baseline is a strong ceiling, and hybrid methods provide targeted gains
- **Source files:** `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv`

#### 4.3.2 Dense Baseline Results
- **Goal:** Characterise the baseline; establish what it does well and where it fails
- **Results:** Dense row from Table 4.7
- **Key interpretation:** NDCG=0.7337, Hit@1=0.6717, Hit@10=0.7811. The 10.9 percentage-point gap between Hit@1 and Hit@10 (Table 4.11) means roughly one in nine questions has the answer in the top-10 but not at rank 1.
- **Source files:** `data/results/retrieval/pure_semantic_dense/metrics.json`

#### 4.3.3 Hybrid Retrieval Results
- **Goal:** Characterise the three hybrid methods; explain `hybrid_type_filtering` == dense
- **Results:** Hybrid rows from Table 4.7; retrieval_by_entity_type and retrieval_by_difficulty breakdowns
- **Key interpretation points:**
  - `hybrid_type_filtering` == dense confirms collection purity; should be framed as a positive validation, not a negative result
  - `hybrid_type_onehop_filtering` provides +0.0014 NDCG; marginal overall but improves `paper_to_authors` question type by boosting graph-connected candidates
  - `hybrid_predicate_aware_filtering` improves Hit@1 by +0.0151 (0.6868 vs 0.6717) and NDCG by +0.0038; the predicate boost helps identify the correct entity when question intent is clear (e.g., task-related questions)
  - Not all question types benefit equally from predicate-aware boosting; some types see NDCG decline when the predicate evidence is ambiguous
- **Source files:** `data/results/retrieval/hybrid_*/metrics.json`

#### 4.3.4 Multi-Representation Fusion Results
- **Goal:** Characterise the RRF methods as exploratory; explain the precision–recall tradeoff
- **Results:** RRF rows from Table 4.7; Table 4.11 (Hit@1 vs Hit@10 gap)
- **Key interpretation points:**
  - `optional_rrf_fusion` trades Hit@1 (-0.0226 vs dense) for Hit@10 gain (+0.0378); the gold entity enters the top-10 more often but is harder to rank first
  - `optional_rrf_symbolic` partially recovers Hit@1 (+0.0125 vs rrf_fusion) while retaining the Hit@10 gain; best NDCG overall (0.7434)
  - The RRF Hit@10=0.8189 means 81.9% of questions have the correct answer in the top-10 — important for post-retrieval recoverability
  - Dataset NDCG improves significantly under RRF (0.382 → 0.465 for `optional_rrf_symbolic`); the fusion helps when title-only representations miss the target in dense search
- **Source files:** `data/results/retrieval/optional_rrf_*/metrics.json`; `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv`

#### 4.3.5 Performance by Entity Type
- **Goal:** Show differential method effectiveness by entity type
- **Results:** Table 4.8; Figure 4.7 (must generate)
- **Key notes:** Use verified values from `retrieval_by_entity_type_ndcg.csv`. Do not invent per-entity Hit@1 values — these are only available in the CSV as NDCG. If Hit@1 per entity type is needed, it must be computed from results.json files.

#### 4.3.6 Performance by Difficulty
- **Goal:** Confirm hard questions as the primary bottleneck; show ceiling effect on easy questions
- **Results:** Table 4.9; Figure 4.8 (must generate)
- **Key notes:** All methods converge near 0.981–0.991 NDCG on easy questions. Hard question NDCG ranges 0.467–0.497. No method lifts hard questions above 0.50 NDCG. This is the key negative finding for RQ2: hybrid methods do not solve the hard-question bottleneck.

#### 4.3.7 Performance by Question Type
- **Goal:** Identify question types that benefit from hybrid/RRF; identify structural failures
- **Results:** Table 4.10; Figure 4.9 heatmap (must generate)
- **Key notes:** Select 4–5 question types for prose discussion: (a) perfect types (model_family_variant, paper_to_author_count), (b) types that improve under RRF (semantic_task_to_dataset, dataset_to_task_membership), (c) structural zeros (tasks_to_dataset, paper_by_author_and_task). Refer to Table 4.10 for the full picture.

#### 4.3.8 Error Analysis of Retrieval Failures
- **Goal:** Classify and quantify failure modes; provide qualitative grounding
- **Results:** Table 4.12 (must generate from results.json); Figure 4.11 (must generate)
- **Notes:** See Section 8 for the full error analysis plan. Do not report example questions unless extracted from the actual questions file.
- **Source files:** `data/results/retrieval/pure_semantic_dense/results.json`; `data/questions/ml_questions_dataset.json`

#### 4.3.9 Summary and Answer to RQ2
- **Goal:** State the direct answer to RQ2
- **Text guidance:** Hybrid symbolic–semantic methods provide modest improvements over the dense baseline, primarily in NDCG (+0.001 to +0.010) and selectively in recall (Hit@10 under RRF). The improvements are largest for dataset entities and for specific question types that align with predicate signals. However, the strong dense baseline — enabled by well-tuned pre-retrieval representations — limits the headroom for hybrid gains. The primary bottleneck is hard-question performance, which no current method resolves.

---

### 4.4 Post-Retrieval Results [PLANNED — no results available]

#### 4.4.1 Candidate Evidence Availability
- **Goal:** Report the available evidence entering the post-retrieval stage from verified data
- **Results:** Hit@10 under best retrieval method (`optional_rrf_symbolic`): 0.8189 (81.89% of questions have the gold entity in the top-10 candidate list). This is the maximum theoretical recall ceiling for any downstream re-ranker operating on 10 candidates.
- **Source files:** `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv`
- **Note:** This sub-section can be written now from retrieval results.

#### 4.4.2 Re-Ranking Results [TODO]
- **Status:** The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is implemented in `src/post_retrieval/`. No results generated.
- **When result file exists:** Report re-ranked Hit@1 and NDCG; compare to pre-re-ranking retrieval Hit@1.
- **Required file:** `data/results/post_retrieval/reranking_results.json` or equivalent

#### 4.4.3 Answer Generation Results [TODO]
- **Status:** TinyLlama-1.1B answer generation pipeline is implemented. No results generated.
- **When result file exists:** Report SAS, ROUGE-L, LLM-judge correct (%). Table 4.13.
- **Required file:** `data/results/post_retrieval/generation_results.json` or equivalent

#### 4.4.4 Qualitative Examples [TODO]
- **Status:** Cannot be written without generated answers.
- **When available:** Include 2–3 question–answer pairs showing correct generation, near-miss, and failure.

#### 4.4.5 Summary [TODO]

---

### 4.5 Cross-Stage Analysis

#### 4.5.1 How Pre-Retrieval Choices Affect Retrieval
- **Goal:** Connect the pre-retrieval ablation findings to the retrieval-stage outcomes
- **Key points:**
  - The dense baseline inherits its strong performance directly from the pre-retrieval representation choices; a different representation (e.g., `dataset_predicate_filtered` instead of `dataset_title_only`) would degrade the baseline significantly
  - The entity-type-specific best representations enable the type-filtering control to produce identical results to the dense baseline — a sign that type filtering adds no incremental information when collections are already entity-pure
  - Predicate-aware hybrid gains are partly attributable to the same predicate signals that made `model_predicate_filtered` and `paper_predicate_filtered` strong in pre-retrieval
- **No new tables or figures needed**

#### 4.5.2 Recoverability and Top-10 Candidate Quality
- **Goal:** Frame the Hit@1–Hit@10 gap as the motivation for re-ranking
- **Results:** Table 4.11; Figure 4.10 (must generate)
- **Key point:** Under `optional_rrf_symbolic`, 81.9% of questions have the gold entity in the top-10; only 66.4% have it at rank 1. The gap of 15.5 percentage points is the theoretical maximum improvement a re-ranker could achieve assuming it perfectly identifies the gold entity from the candidate list.

#### 4.5.3 Main Failure Modes
- **Goal:** Synthesise failure modes from all stages into a unified list
- **Failure categories:**
  1. **Gold absent from top-10** — affects ~18% of questions under best retrieval; root cause: representation text does not contain sufficient query-matching signal
  2. **Metadata sparsity** — primarily affects datasets; sparse MLSea metadata means embedding space is dominated by noise
  3. **Multi-predicate questions** — questions requiring intersection of multiple constraints (e.g., `paper_by_author_and_task`) cannot be resolved by cosine similarity alone
  4. **Set-retrieval question types** — `tasks_to_dataset`, `semantic_task_to_dataset` require matching a set of attributes to a single entity; beyond what dense retrieval handles
  5. **Difficulty scaling** — hard questions fail not because of method choice but because of low query-entity embedding discriminability
- **Source files:** `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv`

---

### 4.6 Final Summary of Findings

- **Goal:** Concise synthesis of RQ1, RQ2, and the post-retrieval outlook
- **Content:**
  - RQ1 answered: entity-type-specific representations are optimal; `enriched_metadata` for papers, `dataset_title_only` for datasets, `model_predicate_filtered` for models
  - RQ2 answered: hybrid methods provide real but small NDCG gains; the dense baseline is strong due to pre-retrieval tuning; hard questions remain unresolved
  - Post-retrieval outlook: 81.9% Hit@10 ceiling provides meaningful headroom for re-ranking; post-retrieval results are pending
  - Main limitation: single-gold-target evaluation underestimates recall for multi-valid-answer questions

---

## 7. Result Interpretation Guidelines

### 7.1 General Principles

- **Separate reporting from explanation.** First state the numbers; then interpret them. Do not blend metric reporting with speculation.
- **Do not overclaim.** Differences below 0.01 NDCG are within measurement noise for 265 questions; phrase them as "marginal" or "negligible."
- **Use hedged language for causal claims.** Write "results suggest" or "this is consistent with" rather than "this proves."
- **Do not compare paper, dataset, and model metrics directly.** The evaluation populations differ in size and difficulty distribution.

### 7.2 Entity-Specific Interpretation Notes

**Why `dataset_title_only` may outperform enriched dataset representations:**
MLSea dataset metadata is frequently sparse. Enriched representations contain incomplete or repeated attribute values that introduce embedding noise without adding discriminative signal. The title alone is the most reliable and semantically dense text fragment for a dataset entity. This finding is consistent with known limitations of knowledge-graph-grounded retrieval when the underlying graph is sparsely populated. Write this as an empirical observation with a structural explanation, not as a general claim that titles always outperform enriched representations.

**Why `enriched_metadata` outperforms simpler strategies for papers:**
Papers in MLSea are text-rich entities. The abstract, author list, tasks, datasets, and implementation links all provide complementary matching signals for different question types. A question asking "which paper uses dataset X" can be matched to an enriched representation that includes the dataset field, but not to a title-only representation. This is supported by the data: `enriched_metadata` has the largest relative gain on task-related paper question types.

**Why `model_predicate_filtered` is dominant:**
Model entities in MLSea are identified primarily through repository links, not natural-language descriptions. Model names are short and may be ambiguous (many models share generic names like "BERT fine-tuned"). Predicate filtering selects the most discriminative structured fields (repository URL, linked paper, task category), producing an embedding that captures the model's functional identity rather than just its name.

### 7.3 Retrieval Interpretation Notes

**Whether hybrid methods improve retrieval:**
The aggregate NDCG improvements are real (verified) but small (+0.001 to +0.010). The correct framing is: hybrid methods provide targeted improvements for specific sub-populations, while the overall aggregate metric is dominated by the already-strong dense baseline. Do not write "hybrid methods fail to improve" — they do improve, selectively. Do not write "hybrid methods substantially outperform dense" — the improvements are marginal in aggregate.

**Why RRF improves Hit@10 but decreases Hit@1:**
RRF combines rankings from multiple representation strategies (e.g., `enriched_metadata` and `predicate_filtered` for papers). Aggregating diverse signals broadens the candidate pool, pulling additional relevant entities into the top-10. However, the fusion of rankings can displace a high-confidence rank-1 prediction from the best single strategy. This is a known precision–recall tradeoff in fusion-based retrieval.

**How Hit@10 relates to post-retrieval recoverability:**
Hit@10 is the ceiling recall for any re-ranker operating on the fixed top-10 candidate list. If Hit@10=0.819, then 18.1% of questions are unrecoverable by re-ranking alone — the gold entity is not in the pool. Post-retrieval evaluation should always compare against this ceiling.

**Limitations from sparse metadata:**
Dataset performance is systematically lower than paper and model performance. This is a structural limitation of the evaluation dataset, not a failure of the approach. Write: "the lower retrieval performance on dataset-type questions reflects the sparser metadata available for dataset entities in the MLSea knowledge graph rather than a fundamental limitation of the representation approach."

**Limitations from single-gold-target evaluation:**
Each question has exactly one gold entity IRI. For questions that may have multiple valid answers (e.g., "which dataset is used for task X?" when multiple datasets are valid), any correctly retrieved alternative entity is penalised as a false positive. Acknowledge this as a conservative evaluation design that may understate recall.

---

## 8. Error Analysis Plan

The error analysis should be grounded in per-question results from `data/results/retrieval/pure_semantic_dense/results.json`. All counts and examples must be derived from actual files.

### 8.1 Failure Categories to Analyse

**Category 1: Gold entity absent from top-10 (Hit@10 = 0)**
- Count questions where the gold entity IRI does not appear in any of the top-10 candidates
- Segment by entity type and question type
- Expected finding: highest absence rate for `tasks_to_dataset` and `paper_by_author_and_task` question types
- Source: `data/results/retrieval/pure_semantic_dense/results.json` — check each question for absence of `target_entity_iri` in top-10 candidates

**Category 2: Gold entity present but ranked low (Hit@1 = 0, Hit@10 = 1)**
- Count questions where the gold entity is in position 2–10 but not position 1
- These are recoverable by re-ranking; quantify this population
- Verified count can be computed as: questions where Hit@10=1 and Hit@1=0
- Source: same file

**Category 3: Semantically similar distractor at rank 1**
- Identify cases where a wrong entity at rank 1 shares high semantic similarity with the gold entity
- This requires manual inspection of a sample (~10–20 cases); do not claim systematic statistics without computing them
- Example: a question targeting paper A may retrieve paper B with the same task, same year, and same author group
- Source: `data/results/retrieval/pure_semantic_dense/top10.json` (or results.json candidate lists)

**Category 4: Question type with structurally zero signal**
- Identify question types where Hit@10 = 0 for all or nearly all questions
- Verified from `retrieval_by_question_type_ndcg.csv`: `tasks_to_dataset` (NDCG=0.0, dense); `title_and_year_filter_to_papers` (NDCG=0.0); `paper_by_author_and_task` (NDCG=0.133)
- Explain the structural reason: these questions require set-intersection or multi-predicate filtering that cosine similarity cannot perform

**Category 5: Metadata sparsity failure**
- Identify dataset questions where all strategies fail (Hit@10 = 0 for `dataset_title_only`)
- These represent questions where even the title signal is insufficient (e.g., generic dataset names)
- Source: `data/results/pre_retrieval_results/dataset_results/dataset_title_only/results.json`

**Category 6: Difficulty scaling**
- Show that hard questions fail at a far higher rate than easy/medium across all methods
- Verified: hard question NDCG ≤ 0.497 for all methods; easy question NDCG ≥ 0.981
- Do not claim all hard questions are unsolvable — `model_predicate_filtered` achieves NDCG=0.943 on hard model questions

### 8.2 Qualitative Example Extraction

Extract 3–5 example questions that clearly illustrate specific failure modes. Do NOT invent examples. For each example:
- Extract the actual question text from `data/questions/ml_questions_dataset.json`
- Show the question type, gold entity IRI, and gold entity display name
- Show what the top-1 retrieved entity was and why it might have scored highly
- Label the failure category

Example extraction query (conceptual):
```
# For tasks_to_dataset NDCG=0 questions:
# 1. Load ml_questions_dataset.json
# 2. Filter to question_type == "tasks_to_dataset"
# 3. Load pure_semantic_dense/results.json
# 4. Cross-reference to find questions with Hit@10=0
# 5. Display the question text, gold IRI, and top-1 retrieved entity
```

### 8.3 Representation Noise Failures

- For paper `predicate_filtered` and model `model_enriched_metadata`, check whether hard questions that succeed under predicate_filtered but fail under enriched_metadata (or vice versa) reveal which predicates add signal vs. noise
- This is an advanced analysis; include only if data supports a clear interpretation

---

## 9. LaTeX Integration Plan

### 9.1 File to Create

All Chapter 4 content should be written to:
```
docs/chapter4_results.tex
```
This file does not yet exist and must be created.

### 9.2 Table Formatting

**Small tables (≤ 8 rows, ≤ 7 columns):** Use standard `table` + `tabular` environment.

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{...}
  \label{tab:label_here}
  \begin{tabular}{llrrrrr}
    \toprule
    ...
    \bottomrule
  \end{tabular}
\end{table}
```

**Medium tables (8–15 rows):** Use `table` with `\footnotesize` font.

**Wide tables (many columns, e.g., question-type × method):** Use `sidewaystable`:
```latex
\begin{sidewaystable}
  \centering
  \footnotesize
  \caption{...}
  \label{tab:retrieval_question_type}
  ...
\end{sidewaystable}
```
Requires `\usepackage{rotating}` in the preamble.

**Very long tables spanning pages:** Use `longtable` from the `longtable` package. Appropriate for Table 4.10 if all 22 question types are included.

### 9.3 Recommended Labels

Use these labels consistently throughout `chapter4_results.tex` and cross-reference them from
other chapters if needed:

| Table | Recommended Label |
|---|---|
| Table 4.1 — Full pre-retrieval comparison | `tab:pre_retrieval_overall` |
| Table 4.2 — Best per entity | `tab:best_representation_by_entity` |
| Table 4.3 — Paper results | `tab:paper_representation_results` |
| Table 4.4 — Dataset results | `tab:dataset_representation_results` |
| Table 4.5 — Model results | `tab:model_representation_results` |
| Table 4.6 — Difficulty breakdown (pre-retrieval) | `tab:pre_retrieval_difficulty` |
| Table 4.7 — Retrieval method comparison | `tab:retrieval_method_comparison` |
| Table 4.8 — Retrieval by entity type | `tab:retrieval_by_entity_type` |
| Table 4.9 — Retrieval by difficulty | `tab:retrieval_by_difficulty` |
| Table 4.10 — Retrieval by question type | `tab:retrieval_by_question_type` |
| Table 4.11 — Hit@1 vs Hit@10 gap | `tab:retrieval_hit_gap` |
| Table 4.12 — Error analysis | `tab:error_analysis` |

| Figure | Recommended Label |
|---|---|
| Figure 4.1 — NDCG paper | `fig:ndcg_paper` |
| Figure 4.2 — Hit@1 paper | `fig:hit1_paper` |
| Figure 4.3 — NDCG dataset | `fig:ndcg_dataset` |
| Figure 4.4 — NDCG model | `fig:ndcg_model` |
| Figure 4.5 — Difficulty breakdown | `fig:representation_difficulty_ndcg` |
| Figure 4.6 — Retrieval method comparison | `fig:retrieval_methods_ndcg` |
| Figure 4.7 — Retrieval by entity type | `fig:retrieval_entity_type_ndcg` |
| Figure 4.8 — Retrieval by difficulty | `fig:retrieval_difficulty_ndcg` |
| Figure 4.9 — Question type heatmap | `fig:question_type_heatmap` |
| Figure 4.10 — Hit@1 vs Hit@10 | `fig:retrieval_hit_gap` |
| Figure 4.11 — Failure mode distribution | `fig:failure_mode_distribution` |

### 9.4 Figure Inclusion

Include existing PDF figures from `data/results/thesis_figures/` using:
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{../data/results/thesis_figures/ndcg_paper.pdf}
  \caption{...}
  \label{fig:ndcg_paper}
\end{figure}
```

For side-by-side figures (e.g., NDCG and Hit@1 for the same entity type):
```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{../data/results/thesis_figures/ndcg_paper.pdf}
    \caption{NDCG}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{../data/results/thesis_figures/hit1_paper.pdf}
    \caption{Hit@1}
  \end{subfigure}
  \caption{Paper representation performance.}
  \label{fig:paper_repr_combined}
\end{figure}
```
Requires `\usepackage{subcaption}`.

### 9.5 Caption Style

- First sentence: states what the figure/table shows.
- Second sentence: highlights the key finding or the most important value.
- Third sentence (optional): notes any important caveat (e.g., absent difficulty bars, question-type evaluation design).
- Do not write captions that merely say "results are shown above."

### 9.6 Number Formatting

Use 4 decimal places for metric values in prose (0.8225); use 4 decimal places in tables. Do not round to 2 decimal places — the differences between methods are in the third and fourth decimal places.

### 9.7 booktabs Requirement

All tables should use `\toprule`, `\midrule`, `\bottomrule` from the `booktabs` package. Do not use horizontal lines inside the table body unless grouping entity types. Use a mid-rule to separate paper/dataset/model groups in Table 4.1.

---

## 10. Claims That Can Be Made Only If Verified

The following claims must be checked against the stated file before writing them in the thesis.

| Claim | File to Verify | Status |
|---|---|---|
| "enriched_metadata is the best paper representation" | `data/results/thesis_tables/best_per_entity.csv` — paper row | **Verified: NDCG=0.8225** |
| "dataset_title_only is the best dataset representation" | `data/results/thesis_tables/best_per_entity.csv` — dataset row | **Verified: NDCG=0.3822** |
| "model_predicate_filtered is the best model representation" | `data/results/thesis_tables/best_per_entity.csv` — model row | **Verified: NDCG=0.8750** |
| "enriched metadata outperforms title_only for papers" | `data/results/thesis_tables/full_comparison.csv` | **Verified: 0.8225 > 0.7346** |
| "dataset_title_only outperforms dataset_enriched_metadata" | `data/results/thesis_tables/full_comparison.csv` | **Verified: 0.3822 > 0.3243** |
| "model_predicate_filtered substantially outperforms model_title_only" | `data/results/thesis_tables/full_comparison.csv` | **Verified: 0.8750 vs 0.4465** |
| "abstract_only is the worst paper representation" | `data/results/thesis_tables/full_comparison.csv` | **Verified: NDCG=0.5438** |
| "hybrid_type_filtering produces identical results to the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_main_comparison.csv` | **Verified: both NDCG=0.7337** |
| "optional_rrf_symbolic achieves the highest overall NDCG" | `data/results/retrieval/summary.csv` | **Verified: NDCG=0.7434** |
| "RRF methods improve Hit@10 over the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` | **Verified: 0.8189 vs 0.7811** |
| "RRF methods reduce Hit@1 compared to the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_precision_recall_tradeoff.csv` | **Verified: fusion 0.6491 < dense 0.6717** |
| "Hard questions yield NDCG below 0.50 across all methods" | `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` | **Verified: hard column max=0.4972** |
| "Easy questions yield NDCG above 0.98 across all methods" | `data/results/retrieval/thesis_tables/retrieval_by_difficulty_ndcg.csv` | **Verified: easy column min=0.9815** |
| "Dataset NDCG improves under RRF symbolic vs dense" | `data/results/retrieval/thesis_tables/retrieval_by_entity_type_ndcg.csv` | **Verified: 0.4645 vs 0.3822** |
| "tasks_to_dataset yields NDCG=0 for the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` | **Verified: NDCG=0.0, n=4** |
| "semantic_task_to_dataset yields NDCG=0.071 for the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` | **Verified: NDCG=0.0714** |
| "paper_by_author_and_task yields NDCG=0.133 for the dense baseline" | `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` | **Verified: NDCG=0.1333** |
| "265 answerable questions are used for quantitative evaluation" | `data/results/retrieval/summary.csv` (evaluated_questions column) | **Verified: 265** |
| "Post-retrieval improves answer quality" | `data/results/post_retrieval/` | **NOT VERIFIABLE — directory does not exist. Do not write this claim.** |
| "Re-ranking improves Hit@1 over retrieval Hit@1" | `data/results/post_retrieval/` | **NOT VERIFIABLE — TODO** |
| "SAS score exceeds [threshold]" | `data/results/post_retrieval/` | **NOT VERIFIABLE — TODO** |

---

## 11. Immediate Next Steps

Prioritised action list for completing Chapter 4:

### Priority 1 — Verify and Consolidate Existing Results (Ready Now)
- [ ] Read `data/results/thesis_tables/full_comparison.csv` and verify all 14 rows are present and complete
- [ ] Read `data/results/retrieval/summary.csv` and verify all 6 method rows are present
- [ ] Read `data/results/retrieval/thesis_tables/retrieval_by_question_type_ndcg.csv` and confirm 22 question types × 6 methods are present
- [ ] Check for question-type coverage gaps (any type with n=0 rows?)

### Priority 2 — Generate Missing Figures (5 new figures needed)
- [ ] **Figure 4.6:** Bar chart of NDCG + Hit@1 by retrieval method — from `retrieval_main_comparison.csv`
- [ ] **Figure 4.7:** NDCG by entity type and method — from `retrieval_by_entity_type_ndcg.csv`
- [ ] **Figure 4.8:** NDCG by difficulty and method — from `retrieval_by_difficulty_ndcg.csv`
- [ ] **Figure 4.9:** NDCG heatmap (question type × method) — from `retrieval_by_question_type_ndcg.csv`
- [ ] **Figure 4.10:** Hit@1 vs Hit@10 with gap — from `retrieval_precision_recall_tradeoff.csv`
- [ ] **Figure 4.11:** Failure mode distribution (must compute from `pure_semantic_dense/results.json`)
- [ ] Save all new figures as both PDF (300 dpi) and PNG to `data/results/thesis_figures/`

### Priority 3 — Generate Error Analysis Table
- [ ] Load `data/results/retrieval/pure_semantic_dense/results.json`
- [ ] Count questions where Hit@10=0 (gold absent); where Hit@10=1 and Hit@1=0 (present but not top-ranked)
- [ ] Segment by entity type and question type
- [ ] Extract 3–5 failure examples with actual question text from `data/questions/ml_questions_dataset.json`
- [ ] Produce Table 4.12

### Priority 4 — Write Chapter 4 LaTeX
- [ ] Create `docs/chapter4_results.tex`
- [ ] Write §4.1 (Evaluation Setup) — no new data needed
- [ ] Write §4.2 (Pre-Retrieval Results) using Tables 4.1–4.6 and Figures 4.1–4.5
- [ ] Write §4.3 (Retrieval Results) using Tables 4.7–4.12 and Figures 4.6–4.11
- [ ] Write §4.4 (Post-Retrieval) — §4.4.1 can be written now; §4.4.2–4.4.5 are TODO placeholders
- [ ] Write §4.5 (Cross-Stage Analysis) — §4.5.1 and §4.5.2 can be written now
- [ ] Write §4.6 (Final Summary)

### Priority 5 — Run Post-Retrieval Pipeline (Required for §4.4)
- [ ] Run `src/post_retrieval/scripts/run_post_retrieval_pipeline.py` to generate re-ranking and answer generation results
- [ ] Verify output lands in `data/results/post_retrieval/`
- [ ] Run `src/post_retrieval/scripts/run_evaluate_generation.py` for SAS, ROUGE-L, LLM-judge metrics
- [ ] Only then write §4.4.2–4.4.5

### Priority 6 — Cross-Check and Proofreading
- [ ] Verify every number in thesis prose against its source CSV
- [ ] Ensure table labels match figure cross-references in text
- [ ] Confirm all figures render correctly in LaTeX (check PDF path, `\includegraphics` width)
- [ ] Check that the post-retrieval TODO sections are clearly marked as pending

---

## Chapter 4 Completion Checklist

### Evaluation Setup
- [ ] Reported: 265 answerable questions used (280 total, 15 unanswerable excluded)
- [ ] Reported: entity distribution (178 paper, 57 dataset, 30 model questions)
- [ ] Reported: 35 question types, difficulty distribution stated
- [ ] Metrics defined by reference to Chapter 3 §3.3 (no re-derivation)

### Pre-Retrieval Results
- [ ] Table 4.1 generated from `full_comparison.csv` and inserted in LaTeX
- [ ] Table 4.2 generated from `best_per_entity.csv` and inserted
- [ ] Table 4.3 (paper-only), Table 4.4 (dataset-only), Table 4.5 (model-only) generated
- [ ] Table 4.6 (difficulty breakdown) generated from `difficulty_breakdown.csv`
- [ ] Figure 4.1 (ndcg_paper.pdf) included — label `fig:ndcg_paper`
- [ ] Figure 4.2 (hit1_paper.pdf) included — label `fig:hit1_paper`
- [ ] Figure 4.3 (ndcg_dataset.pdf) included — label `fig:ndcg_dataset`
- [ ] Figure 4.4 (ndcg_model.pdf) included — label `fig:ndcg_model`
- [ ] Figure 4.5 (best_repr_difficulty_breakdown_ndcg.pdf) included — label `fig:representation_difficulty_ndcg`
- [ ] RQ1 explicitly answered in §4.2.7

### Retrieval Results
- [ ] Table 4.7 generated from `retrieval_main_comparison.csv` and inserted
- [ ] Table 4.8 from `retrieval_by_entity_type_ndcg.csv` generated
- [ ] Table 4.9 from `retrieval_by_difficulty_ndcg.csv` generated
- [ ] Table 4.10 from `retrieval_by_question_type_ndcg.csv` generated (full or abbreviated)
- [ ] Table 4.11 from `retrieval_precision_recall_tradeoff.csv` generated
- [ ] Table 4.12 (error analysis) computed from `results.json` files and inserted
- [ ] Figure 4.6 (retrieval method comparison bar) generated and inserted
- [ ] Figure 4.7 (entity type breakdown) generated and inserted
- [ ] Figure 4.8 (difficulty breakdown) generated and inserted
- [ ] Figure 4.9 (question type heatmap) generated and inserted
- [ ] Figure 4.10 (Hit@1 vs Hit@10 gap) generated and inserted
- [ ] Figure 4.11 (failure mode distribution) generated and inserted
- [ ] RQ2 explicitly answered in §4.3.9

### Post-Retrieval Results
- [ ] §4.4.1 written (candidate evidence availability from retrieval Hit@10)
- [ ] §4.4.2–4.4.5 clearly marked TODO pending pipeline execution — OR completed after running post-retrieval pipeline

### Cross-Stage and Summary
- [ ] §4.5 written (cross-stage analysis — §4.5.1 and §4.5.2 can be done now)
- [ ] §4.6 written (final summary)

### LaTeX Integration
- [ ] `docs/chapter4_results.tex` created
- [ ] All tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- [ ] All figures use PDF version from `data/results/thesis_figures/`
- [ ] All labels follow the naming convention in Section 9.3
- [ ] Wide tables use `sidewaystable` or `longtable` as appropriate
- [ ] All numerical claims in prose verified against source CSV files

### Claims Verification
- [ ] No post-retrieval claims written until result files exist
- [ ] No numbers stated in prose that are not present in a verified source file
- [ ] All delta/improvement claims compared explicitly to the dense baseline
