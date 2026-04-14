# Post-retrieval

Post-retrieval remains in the repository for later work, but it is not the active focus of the local workflow.

The active local pipeline currently stops after retrieval evaluation and exports `data/retrieval_results/paper_results/{representation}/top10.json` for future post-retrieval consumption.

Note: The old evaluation scaffold (`src/post_retrieval/evaluation/`) has been archived. Evaluation logic now lives within each entity pipeline under `src/pre_retrieval/shared/evaluate_retrieval.py`.
