from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pre_retrieval.datasets.chunking.build_dataset_enriched_metadata import (
    build_dataset_enriched_metadata_text,
)
from src.pre_retrieval.datasets.chunking.build_dataset_representations import (
    SUPPORTED_DATASET_REPRESENTATIONS,
    build_dataset_representations,
)


DEFAULT_CONFIG = {
    "max_characters": 2400,
    "title_max_characters": 512,
    "description_max_characters": 600,
    "list_item_limit": 8,
    "list_value_max_characters": 120,
    "related_paper_limit": 6,
    "implementation_limit": 4,
    "linked_entity_limit": 6,
}


RICH_RECORD = {
    "dataset_id": "http://w3id.org/mlsea/pwc/dataset/CIFAR-10",
    "dataset_uri": "http://w3id.org/mlsea/pwc/dataset/CIFAR-10",
    "label": "CIFAR-10",
    "title": "CIFAR-10",
    "description": "A well-known image classification benchmark.",
    "issued_year": "2009",
    "keywords": ["computer vision", "benchmark", "image classification"],
    "tasks": ["Image Classification", "Object Recognition"],
    "related_papers": ["Learning Multiple Layers of Features from Tiny Images"],
    "related_implementations": ["https://github.com/example/cifar10"],
    "linked_entities": [
        {"object_label": "Image Classification", "predicate_label": "hasTaskType"},
        {"object_label": "CIFAR-100", "predicate_label": "relatedTo"},
    ],
}

SPARSE_RECORD = {
    "dataset_id": "http://w3id.org/mlsea/pwc/dataset/TinyDS",
    "label": "TinyDS",
}


class TestDatasetEnrichedMetadataBuilder(unittest.TestCase):
    def test_rich_record_includes_all_sections(self) -> None:
        text = build_dataset_enriched_metadata_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIn("Dataset: CIFAR-10", text)
        self.assertIn("Year: 2009", text)
        self.assertIn("Tasks:", text)
        self.assertIn("Keywords:", text)
        self.assertIn("Related Papers:", text)
        self.assertIn("Implementations:", text)
        self.assertIn("Linked Entities:", text)
        self.assertIn("Description:", text)

    def test_sparse_record_returns_title_only(self) -> None:
        text = build_dataset_enriched_metadata_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIn("Dataset: TinyDS", text)
        self.assertNotIn("Year:", text)
        self.assertNotIn("Tasks:", text)

    def test_empty_record_returns_empty(self) -> None:
        text = build_dataset_enriched_metadata_text({}, DEFAULT_CONFIG)
        self.assertEqual(text, "")

    def test_respects_max_characters(self) -> None:
        cfg = {**DEFAULT_CONFIG, "max_characters": 50}
        text = build_dataset_enriched_metadata_text(RICH_RECORD, cfg)
        self.assertLessEqual(len(text), 60)  # allow small truncation marker

    def test_no_empty_sections(self) -> None:
        record = {**RICH_RECORD, "description": "", "keywords": []}
        text = build_dataset_enriched_metadata_text(record, DEFAULT_CONFIG)
        self.assertNotIn("Keywords:", text)
        self.assertNotIn("Description:", text)


class TestDatasetEnrichedMetadataRegistered(unittest.TestCase):
    def test_representation_in_supported_list(self) -> None:
        self.assertIn("dataset_enriched_metadata", SUPPORTED_DATASET_REPRESENTATIONS)

    def test_build_produces_output(self) -> None:
        records = [RICH_RECORD, SPARSE_RECORD]
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_path = Path(tmp_dir) / "datasets_master.jsonl"
            output_dir = Path(tmp_dir) / "representations"
            with records_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

            counts = build_dataset_representations(
                records_path=records_path,
                output_dir=output_dir,
                representation_types=["dataset_enriched_metadata"],
                representation_config_map={"dataset_enriched_metadata": DEFAULT_CONFIG},
            )

            self.assertEqual(counts["dataset_enriched_metadata"], 2)
            output_file = output_dir / "dataset_enriched_metadata.jsonl"
            self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
