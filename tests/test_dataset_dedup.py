from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pre_retrieval.datasets.raw.build_dataset_records import (
    deduplicate_dataset_map,
    make_dataset_accumulator,
    _merge_accumulators,
)


class TestDeduplicateDatasetMap(unittest.TestCase):
    """Verify that dataset accumulators with URL-encoding variants are merged."""

    @staticmethod
    def _acc_with_triple(uri: str, predicate: str, obj: str, is_literal: bool = True) -> dict:
        acc = make_dataset_accumulator(uri)
        acc["triples"].append({"predicate": predicate, "object": obj, "is_literal": is_literal})
        acc["raw_predicates"].add(predicate)
        return acc

    def test_merges_url_encoded_duplicates(self) -> None:
        encoded_uri = "http://w3id.org/mlsea/pwc/dataset/Geometric%20Pose%20Affordance"
        plain_uri = "http://w3id.org/mlsea/pwc/dataset/Geometric Pose Affordance"

        dataset_map = {
            encoded_uri: self._acc_with_triple(encoded_uri, "http://schema.org/name", "Geometric Pose Affordance"),
            plain_uri: self._acc_with_triple(plain_uri, "http://schema.org/description", "A detailed description"),
        }

        deduped_map, merged_count = deduplicate_dataset_map(dataset_map)

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(deduped_map), 1)
        # The surviving accumulator should have triples from both
        surviving = next(iter(deduped_map.values()))
        self.assertEqual(len(surviving["triples"]), 2)

    def test_no_duplicates_passes_through(self) -> None:
        uri_a = "http://w3id.org/mlsea/pwc/dataset/DatasetA"
        uri_b = "http://w3id.org/mlsea/pwc/dataset/DatasetB"

        dataset_map = {
            uri_a: self._acc_with_triple(uri_a, "http://schema.org/name", "A"),
            uri_b: self._acc_with_triple(uri_b, "http://schema.org/name", "B"),
        }

        deduped_map, merged_count = deduplicate_dataset_map(dataset_map)

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(deduped_map), 2)

    def test_merge_accumulators_combines_fields(self) -> None:
        target = make_dataset_accumulator("http://example.org/ds1")
        target["triples"].append({"predicate": "p1", "object": "o1", "is_literal": True})
        target["raw_predicates"].add("p1")
        target["referenced_nodes"].add("http://ref/1")

        source = make_dataset_accumulator("http://example.org/ds1_dup")
        source["triples"].append({"predicate": "p2", "object": "o2", "is_literal": True})
        source["raw_predicates"].add("p2")
        source["referenced_nodes"].add("http://ref/2")

        _merge_accumulators(target, source)

        self.assertEqual(len(target["triples"]), 2)
        self.assertEqual(target["raw_predicates"], {"p1", "p2"})
        self.assertEqual(target["referenced_nodes"], {"http://ref/1", "http://ref/2"})


class TestDefensiveDedupInRepresentations(unittest.TestCase):
    """Verify that build_dataset_representations de-duplicates input records."""

    def test_dedup_removes_duplicate_dataset_ids(self) -> None:
        from src.pre_retrieval.datasets.chunking.build_dataset_representations import build_dataset_representations

        records = [
            {"dataset_id": "http://w3id.org/mlsea/pwc/dataset/Tox21", "label": "Tox21", "title": "Tox21", "description": "First record"},
            {"dataset_id": "http://w3id.org/mlsea/pwc/dataset/Tox21", "label": "Tox21", "title": "Tox21", "description": "Duplicate record"},
            {"dataset_id": "http://w3id.org/mlsea/pwc/dataset/Other", "label": "Other", "title": "Other", "description": "Unique record"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            records_path = Path(tmp_dir) / "datasets_master.jsonl"
            output_dir = Path(tmp_dir) / "representations"
            with records_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

            counts = build_dataset_representations(
                records_path=records_path,
                output_dir=output_dir,
                representation_types=["dataset_title_only"],
                representation_config_map={"dataset_title_only": {"max_characters": 512}},
            )

            self.assertEqual(counts["dataset_title_only"], 2)  # 2 unique, not 3


if __name__ == "__main__":
    unittest.main()
