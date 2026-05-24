# Retrieval Stage — Method Comparison

| method | evaluated_questions | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG |
|---|---|---|---|---|---|---|
| pure_semantic_dense | 500 | 0.378 | 0.544 | 0.598 | 0.4496 | 0.4854 |
| hybrid_type_filtering | 500 | 0.378 | 0.544 | 0.598 | 0.4496 | 0.4854 |
| hybrid_type_onehop_filtering | 500 | 0.38 | 0.546 | 0.598 | 0.451 | 0.4865 |
| hybrid_predicate_aware_filtering | 500 | 0.382 | 0.556 | 0.598 | 0.4562 | 0.4907 |
| optional_rrf_fusion | 500 | 0.378 | 0.558 | 0.618 | 0.4576 | 0.4962 |
| optional_rrf_symbolic | 500 | 0.396 | 0.574 | 0.618 | 0.474 | 0.5091 |
