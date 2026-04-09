from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.pre_retrieval.evaluation.aggregate_results import aggregate_result_files
from src.pre_retrieval.evaluation.evaluate_retrieval import build_evaluation_payload
from src.pre_retrieval.utils import build_item_id, save_json


TOP_K_VALUES = (1, 5, 10)
PAPER_1 = "http://w3id.org/mlsea/pwc/scientificWork/Paper%201"
PAPER_2 = "http://w3id.org/mlsea/pwc/scientificWork/Paper%202"
PAPER_3 = "http://w3id.org/mlsea/pwc/scientificWork/Paper%203"
DATASET_1 = "http://w3id.org/mlsea/pwc/dataset/Dataset%201"


def _result(rank: int, paper_id: str, score: float) -> dict[str, object]:
    return {
        "rank": rank,
        "item_id": build_item_id("title_only", paper_id),
        "paper_id": paper_id,
        "title": paper_id.rsplit("/", 1)[-1],
        "representation_type": "title_only",
        "text_length_chars": 10,
        "distance": 1.0 - score,
        "score": score,
        "source_text": paper_id.rsplit("/", 1)[-1],
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def test_build_evaluation_payload_segments_questions_and_unanswerable(self) -> None:
        questions = [
            {
                "id": "q_easy",
                "question": "easy paper question",
                "difficulty": "easy",
                "category": "paper",
                "target_entity_iri": PAPER_1,
                "is_answerable": True,
            },
            {
                "id": "q_medium",
                "question": "medium multihop question",
                "difficulty": "medium",
                "category": "multihop",
                "target_entity_iri": PAPER_2,
                "is_answerable": True,
            },
            {
                "id": "q_hard_dataset",
                "question": "hard dataset question",
                "difficulty": "hard",
                "category": "dataset",
                "target_entity_iri": DATASET_1,
                "is_answerable": True,
            },
            {
                "id": "q_hard_semantic",
                "question": "hard semantic question",
                "difficulty": "hard",
                "category": "semantic",
                "target_entity_iri": PAPER_3,
                "is_answerable": True,
            },
            {
                "id": "q_unanswerable",
                "question": "unanswerable question",
                "difficulty": "medium",
                "category": "unanswerable",
                "target_entity_iri": None,
                "is_answerable": False,
            },
        ]
        evaluated_questions = [questions[0], questions[1], questions[3]]
        retrieval_results = [
            [_result(1, PAPER_1, 0.95), _result(2, PAPER_2, 0.75)],
            [_result(1, PAPER_1, 0.88), _result(2, PAPER_2, 0.77)],
            [_result(1, PAPER_1, 0.60), _result(2, PAPER_2, 0.50)],
        ]

        payload, top10_payload = build_evaluation_payload(
            representation_type="title_only",
            collection_name="papers_title_only",
            records_path=Path("/tmp/papers_subset.jsonl"),
            embedder_type="sentence_transformer",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k_values=TOP_K_VALUES,
            all_questions=questions,
            evaluated_questions=evaluated_questions,
            retrieval_results=retrieval_results,
            matched_item_ids=[
                build_item_id("title_only", PAPER_1),
                build_item_id("title_only", PAPER_2),
            ],
            collection_size=42,
            record_index={},
            abstention_score_threshold=0.4,
            unanswerable_results=[[_result(1, PAPER_1, 0.35)]],
        )

        self.assertAlmostEqual(payload["metrics"]["Hit@1"], 1.0 / 3.0)
        self.assertAlmostEqual(payload["metrics"]["Hit@5"], 2.0 / 3.0)
        self.assertAlmostEqual(payload["metrics"]["MRR"], 0.5)
        self.assertAlmostEqual(payload["metrics"]["NDCG"], (1.0 + 1.0 / math.log2(3)) / 3.0)
        self.assertEqual(payload["diagnostics"]["total_questions"], 5)
        self.assertEqual(payload["diagnostics"]["answerable_questions"], 4)
        self.assertEqual(payload["diagnostics"]["evaluated_questions"], 3)
        self.assertEqual(payload["diagnostics"]["skipped_non_paper_targets"], 1)
        self.assertEqual(payload["diagnostics"]["total_unanswerable_questions"], 1)
        self.assertEqual(payload["diagnostics"]["counts_by_difficulty"]["easy"]["total_questions"], 1)
        self.assertEqual(payload["diagnostics"]["counts_by_difficulty"]["medium"]["total_questions"], 2)
        self.assertEqual(payload["diagnostics"]["counts_by_difficulty"]["hard"]["total_questions"], 2)
        self.assertEqual(payload["metrics_by_difficulty"]["easy"]["Hit@1"], 1.0)
        self.assertEqual(payload["metrics_by_difficulty"]["medium"]["evaluated_questions"], 1)
        self.assertEqual(payload["metrics_by_difficulty"]["hard"]["skipped_non_paper_targets"], 1)
        self.assertEqual(payload["metrics_by_category"]["multihop"]["Hit@5"], 1.0)
        self.assertEqual(payload["metrics_by_category"]["dataset"]["evaluated_questions"], 0)
        self.assertEqual(payload["metrics_by_category"]["semantic"]["Hit@10"], 0.0)
        self.assertEqual(payload["metrics_by_category"]["unanswerable"]["skipped_unanswerable"], 1)
        self.assertAlmostEqual(payload["metrics_by_category"]["unanswerable"]["unanswerable_rejection_rate"], 1.0)
        self.assertAlmostEqual(payload["metrics_by_category"]["unanswerable"]["false_accept_rate"], 0.0)
        self.assertEqual(len(payload["per_question"]), 3)
        self.assertEqual(payload["per_question"][1]["difficulty"], "medium")
        self.assertEqual(payload["per_question"][1]["category"], "multihop")
        self.assertEqual(len(top10_payload["entries"]), 3)

    def test_build_evaluation_payload_reports_unanswerable_false_accept_rate(self) -> None:
        questions = [
            {
                "id": "q_unanswerable_low",
                "question": "unanswerable low score",
                "difficulty": "easy",
                "category": "unanswerable",
                "target_entity_iri": None,
                "is_answerable": False,
            },
            {
                "id": "q_unanswerable_high",
                "question": "unanswerable high score",
                "difficulty": "hard",
                "category": "unanswerable",
                "target_entity_iri": None,
                "is_answerable": False,
            },
        ]

        payload, _ = build_evaluation_payload(
            representation_type="title_only",
            collection_name="papers_title_only",
            records_path=Path("/tmp/papers_subset.jsonl"),
            embedder_type="sentence_transformer",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k_values=TOP_K_VALUES,
            all_questions=questions,
            evaluated_questions=[],
            retrieval_results=[],
            matched_item_ids=[],
            collection_size=42,
            record_index={},
            abstention_score_threshold=0.4,
            unanswerable_results=[
                [_result(1, PAPER_1, 0.35)],
                [_result(1, PAPER_1, 0.75)],
            ],
        )

        self.assertAlmostEqual(payload["diagnostics"]["abstention"]["unanswerable_rejection_rate"], 0.5)
        self.assertAlmostEqual(payload["diagnostics"]["abstention"]["false_accept_rate"], 0.5)
        self.assertAlmostEqual(payload["metrics_by_category"]["unanswerable"]["unanswerable_rejection_rate"], 0.5)
        self.assertAlmostEqual(payload["metrics_by_category"]["unanswerable"]["false_accept_rate"], 0.5)

    def test_aggregate_result_files_writes_segment_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            for representation, hit_at_1 in (("title_only", 0.3), ("abstract_only", 0.1)):
                representation_dir = results_dir / representation
                save_json(
                    {
                        "representation_type": representation,
                        "metrics": {
                            "Hit@1": hit_at_1,
                            "Hit@5": 0.6,
                            "Hit@10": 0.7,
                            "MRR": 0.5,
                            "NDCG": 0.55,
                        },
                        "metrics_by_difficulty": {
                            "easy": {
                                "total_questions": 1,
                                "answerable_questions": 1,
                                "paper_target_questions": 1,
                                "evaluated_questions": 1,
                                "skipped_questions": 0,
                                "skipped_non_paper_targets": 0,
                                "skipped_unanswerable": 0,
                                "Hit@1": hit_at_1,
                                "Hit@5": 1.0,
                                "Hit@10": 1.0,
                                "MRR": hit_at_1,
                                "NDCG": hit_at_1,
                            },
                            "medium": {
                                "total_questions": 1,
                                "answerable_questions": 0,
                                "paper_target_questions": 0,
                                "evaluated_questions": 0,
                                "skipped_questions": 1,
                                "skipped_non_paper_targets": 0,
                                "skipped_unanswerable": 1,
                                "Hit@1": 0.0,
                                "Hit@5": 0.0,
                                "Hit@10": 0.0,
                                "MRR": 0.0,
                                "NDCG": 0.0,
                            },
                            "hard": {
                                "total_questions": 1,
                                "answerable_questions": 1,
                                "paper_target_questions": 0,
                                "evaluated_questions": 0,
                                "skipped_questions": 1,
                                "skipped_non_paper_targets": 1,
                                "skipped_unanswerable": 0,
                                "Hit@1": 0.0,
                                "Hit@5": 0.0,
                                "Hit@10": 0.0,
                                "MRR": 0.0,
                                "NDCG": 0.0,
                            },
                        },
                        "metrics_by_category": {
                            "multihop": {
                                "total_questions": 1,
                                "answerable_questions": 1,
                                "paper_target_questions": 1,
                                "evaluated_questions": 1,
                                "skipped_questions": 0,
                                "skipped_non_paper_targets": 0,
                                "skipped_unanswerable": 0,
                                "Hit@1": hit_at_1,
                                "Hit@5": 1.0,
                                "Hit@10": 1.0,
                                "MRR": hit_at_1,
                                "NDCG": hit_at_1,
                            },
                            "unanswerable": {
                                "total_questions": 1,
                                "answerable_questions": 0,
                                "paper_target_questions": 0,
                                "evaluated_questions": 0,
                                "skipped_questions": 1,
                                "skipped_non_paper_targets": 0,
                                "skipped_unanswerable": 1,
                                "Hit@1": 0.0,
                                "Hit@5": 0.0,
                                "Hit@10": 0.0,
                                "MRR": 0.0,
                                "NDCG": 0.0,
                                "unanswerable_rejection_rate": 1.0,
                                "false_accept_rate": 0.0,
                            },
                        },
                    },
                    representation_dir / "results.json",
                )

            summary = aggregate_result_files(
                output_dir=results_dir,
                representation_order=["title_only", "abstract_only"],
            )

            self.assertEqual([row["representation"] for row in summary["rows"]], ["title_only", "abstract_only"])
            difficulty_segments = summary["summary_by_difficulty"]["segments"]
            self.assertEqual(list(difficulty_segments), ["easy", "medium", "hard"])
            self.assertEqual(
                [row["representation"] for row in difficulty_segments["easy"]["rows"]],
                ["title_only", "abstract_only"],
            )
            category_segments = summary["summary_by_category"]["segments"]
            self.assertEqual(list(category_segments), ["multihop", "unanswerable"])
            self.assertAlmostEqual(
                category_segments["unanswerable"]["rows"][0]["unanswerable_rejection_rate"],
                1.0,
            )
            self.assertTrue((results_dir / "summary.json").exists())
            self.assertTrue((results_dir / "summary.md").exists())
            self.assertTrue((results_dir / "summary.csv").exists())
            self.assertTrue((results_dir / "summary_by_difficulty.json").exists())
            self.assertTrue((results_dir / "summary_by_category.json").exists())


if __name__ == "__main__":
    unittest.main()
