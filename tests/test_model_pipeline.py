from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pre_retrieval.models.chunking.build_model_title_only import build_model_title_only_text
from src.pre_retrieval.models.chunking.build_model_metadata import build_model_metadata_text
from src.pre_retrieval.models.chunking.build_model_predicate_filtered import build_model_predicate_filtered_text
from src.pre_retrieval.models.chunking.build_model_enriched_metadata import build_model_enriched_metadata_text
from src.pre_retrieval.models.chunking.build_model_representations import (
    SUPPORTED_MODEL_REPRESENTATIONS,
    build_model_representations,
)
from src.pre_retrieval.models.raw.build_model_records import (
    deduplicate_model_map,
    make_model_accumulator,
    _merge_accumulators,
)


DEFAULT_CONFIG = {
    "max_characters": 2400,
    "title_max_characters": 512,
    "description_max_characters": 600,
    "list_item_limit": 8,
    "list_value_max_characters": 120,
    "related_paper_limit": 6,
    "implementation_limit": 4,
    "dataset_limit": 6,
    "linked_entity_limit": 6,
}


RICH_RECORD = {
    "model_id": "http://w3id.org/mlsea/pwc/model/ResNet-50",
    "model_uri": "http://w3id.org/mlsea/pwc/model/ResNet-50",
    "label": "ResNet-50",
    "title": "ResNet-50",
    "description": "A deep residual network with 50 layers.",
    "issued_year": "2015",
    "keywords": ["deep learning", "residual networks", "image classification"],
    "tasks": ["Image Classification", "Object Detection"],
    "datasets": ["ImageNet", "COCO"],
    "related_papers": ["Deep Residual Learning for Image Recognition"],
    "related_implementations": ["https://github.com/pytorch/vision"],
    "runs": ["run_123"],
    "metrics": ["Top-1 Accuracy"],
    "linked_entities": [
        {"object_label": "Image Classification", "predicate_label": "hasTaskType"},
        {"object_label": "ImageNet", "predicate_label": "usesDataset"},
    ],
}

SPARSE_RECORD = {
    "model_id": "http://w3id.org/mlsea/pwc/model/TinyModel",
    "label": "TinyModel",
}


class TestModelTitleOnly(unittest.TestCase):
    def test_returns_label(self) -> None:
        text = build_model_title_only_text(RICH_RECORD, {"max_characters": 512})
        self.assertEqual(text, "ResNet-50")

    def test_empty_record(self) -> None:
        text = build_model_title_only_text({}, {"max_characters": 512})
        self.assertEqual(text, "")


class TestModelMetadata(unittest.TestCase):
    def test_rich_record(self) -> None:
        text = build_model_metadata_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: ResNet-50", text)
        self.assertIn("Year: 2015", text)
        self.assertIn("Tasks:", text)
        self.assertIn("Datasets:", text)

    def test_sparse_record(self) -> None:
        text = build_model_metadata_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: TinyModel", text)
        self.assertNotIn("Year:", text)


class TestModelPredicateFiltered(unittest.TestCase):
    def test_rich_record(self) -> None:
        text = build_model_predicate_filtered_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIsNotNone(text)
        self.assertIn("Model: ResNet-50", text)

    def test_sparse_record_filtered_out(self) -> None:
        text = build_model_predicate_filtered_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIsNone(text)


class TestModelEnrichedMetadata(unittest.TestCase):
    def test_rich_record_includes_all_sections(self) -> None:
        text = build_model_enriched_metadata_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: ResNet-50", text)
        self.assertIn("Year: 2015", text)
        self.assertIn("Tasks:", text)
        self.assertIn("Datasets:", text)
        self.assertIn("Keywords:", text)
        self.assertIn("Related Papers:", text)
        self.assertIn("Implementations:", text)
        self.assertIn("Metrics:", text)
        self.assertIn("Linked Entities:", text)
        self.assertIn("Description:", text)

    def test_sparse_record(self) -> None:
        text = build_model_enriched_metadata_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: TinyModel", text)
        self.assertNotIn("Year:", text)

    def test_empty_record(self) -> None:
        text = build_model_enriched_metadata_text({}, DEFAULT_CONFIG)
        self.assertEqual(text, "")

    def test_respects_max_characters(self) -> None:
        cfg = {**DEFAULT_CONFIG, "max_characters": 50}
        text = build_model_enriched_metadata_text(RICH_RECORD, cfg)
        self.assertLessEqual(len(text), 60)

    def test_no_empty_sections(self) -> None:
        record = {**RICH_RECORD, "description": "", "keywords": []}
        text = build_model_enriched_metadata_text(record, DEFAULT_CONFIG)
        self.assertNotIn("Keywords:", text)
        self.assertNotIn("Description:", text)


class TestModelRepresentationsRegistered(unittest.TestCase):
    def test_all_representations_in_supported_list(self) -> None:
        self.assertIn("model_title_only", SUPPORTED_MODEL_REPRESENTATIONS)
        self.assertIn("model_metadata", SUPPORTED_MODEL_REPRESENTATIONS)
        self.assertIn("model_predicate_filtered", SUPPORTED_MODEL_REPRESENTATIONS)
        self.assertIn("model_enriched_metadata", SUPPORTED_MODEL_REPRESENTATIONS)

    def test_build_produces_output(self) -> None:
        records = [RICH_RECORD, SPARSE_RECORD]
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_path = Path(tmp_dir) / "models_master.jsonl"
            output_dir = Path(tmp_dir) / "representations"
            with records_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

            counts = build_model_representations(
                records_path=records_path,
                output_dir=output_dir,
                representation_types=["model_title_only", "model_enriched_metadata"],
                representation_config_map={
                    "model_title_only": {"max_characters": 512},
                    "model_enriched_metadata": DEFAULT_CONFIG,
                },
            )

            self.assertEqual(counts["model_title_only"], 2)
            self.assertEqual(counts["model_enriched_metadata"], 2)
            self.assertTrue((output_dir / "model_title_only.jsonl").exists())
            self.assertTrue((output_dir / "model_enriched_metadata.jsonl").exists())

    def test_predicate_filtered_drops_sparse(self) -> None:
        records = [RICH_RECORD, SPARSE_RECORD]
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_path = Path(tmp_dir) / "models_master.jsonl"
            output_dir = Path(tmp_dir) / "representations"
            with records_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

            counts = build_model_representations(
                records_path=records_path,
                output_dir=output_dir,
                representation_types=["model_predicate_filtered"],
                representation_config_map={"model_predicate_filtered": DEFAULT_CONFIG},
            )

            self.assertEqual(counts["model_predicate_filtered"], 1)  # sparse record filtered


class TestModelDedup(unittest.TestCase):
    @staticmethod
    def _acc_with_triple(uri: str, predicate: str, obj: str, is_literal: bool = True) -> dict:
        acc = make_model_accumulator(uri)
        acc["triples"].append({"predicate": predicate, "object": obj, "is_literal": is_literal})
        acc["raw_predicates"].add(predicate)
        return acc

    def test_merges_url_encoded_duplicates(self) -> None:
        encoded_uri = "http://w3id.org/mlsea/pwc/model/My%20Model"
        plain_uri = "http://w3id.org/mlsea/pwc/model/My Model"

        model_map = {
            encoded_uri: self._acc_with_triple(encoded_uri, "http://schema.org/name", "My Model"),
            plain_uri: self._acc_with_triple(plain_uri, "http://schema.org/description", "A description"),
        }

        deduped_map, merged_count = deduplicate_model_map(model_map)

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(deduped_map), 1)
        surviving = next(iter(deduped_map.values()))
        self.assertEqual(len(surviving["triples"]), 2)

    def test_no_duplicates_passes_through(self) -> None:
        uri_a = "http://w3id.org/mlsea/pwc/model/ModelA"
        uri_b = "http://w3id.org/mlsea/pwc/model/ModelB"

        model_map = {
            uri_a: self._acc_with_triple(uri_a, "http://schema.org/name", "A"),
            uri_b: self._acc_with_triple(uri_b, "http://schema.org/name", "B"),
        }

        deduped_map, merged_count = deduplicate_model_map(model_map)

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(deduped_map), 2)

    def test_merge_accumulators_combines_fields(self) -> None:
        target = make_model_accumulator("http://example.org/m1")
        target["triples"].append({"predicate": "p1", "object": "o1", "is_literal": True})
        target["raw_predicates"].add("p1")
        target["referenced_nodes"].add("http://ref/1")

        source = make_model_accumulator("http://example.org/m1_dup")
        source["triples"].append({"predicate": "p2", "object": "o2", "is_literal": True})
        source["raw_predicates"].add("p2")
        source["referenced_nodes"].add("http://ref/2")

        _merge_accumulators(target, source)

        self.assertEqual(len(target["triples"]), 2)
        self.assertEqual(target["raw_predicates"], {"p1", "p2"})
        self.assertEqual(target["referenced_nodes"], {"http://ref/1", "http://ref/2"})


class TestDefensiveDedupInModelRepresentations(unittest.TestCase):
    def test_dedup_removes_duplicate_model_ids(self) -> None:
        records = [
            {"model_id": "http://w3id.org/mlsea/pwc/model/X", "label": "X", "title": "X", "description": "First"},
            {"model_id": "http://w3id.org/mlsea/pwc/model/X", "label": "X", "title": "X", "description": "Dup"},
            {"model_id": "http://w3id.org/mlsea/pwc/model/Y", "label": "Y", "title": "Y", "description": "Unique"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            records_path = Path(tmp_dir) / "models_master.jsonl"
            output_dir = Path(tmp_dir) / "representations"
            with records_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

            counts = build_model_representations(
                records_path=records_path,
                output_dir=output_dir,
                representation_types=["model_title_only"],
                representation_config_map={"model_title_only": {"max_characters": 512}},
            )

            self.assertEqual(counts["model_title_only"], 2)  # 2 unique, not 3


if __name__ == "__main__":
    unittest.main()
