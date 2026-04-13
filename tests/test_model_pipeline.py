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
from src.pre_retrieval.models.chunking.model_graph_helpers import (
    extract_neighbor_labels,
    extract_repo_names,
    extract_repo_urls,
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
    "linked_entity_limit": 6,
}


# A realistic model record: most top-level lists empty, data in linked_entities
RICH_RECORD = {
    "model_id": "http://w3id.org/mlsea/pwc/model/ResNet-50",
    "model_uri": "http://w3id.org/mlsea/pwc/model/ResNet-50",
    "label": "ResNet-50",
    "title": "ResNet-50",
    "description": "",
    "issued_year": "",
    "keywords": [],
    "tasks": [],
    "datasets": [],
    "related_papers": [],
    "related_implementations": [],
    "runs": [],
    "metrics": [],
    "linked_entities": [
        {
            "predicate": "http://schema.org/codeRepository",
            "predicate_label": "codeRepository",
            "object_uri": "https://github.com/pytorch/vision",
            "object_label": "https://github.com/pytorch/vision",
            "object_types": [],
        },
        {
            "predicate": "http://schema.org/codeRepository",
            "predicate_label": "codeRepository",
            "object_uri": "https://github.com/keras-team/keras-applications",
            "object_label": "https://github.com/keras-team/keras-applications",
            "object_types": [],
        },
        {
            "predicate": "http://w3id.org/mlso/relatedTo",
            "predicate_label": "relatedTo",
            "object_uri": "http://w3id.org/mlsea/pwc/method/residual-connections",
            "object_label": "Residual Connections",
            "object_types": [],
        },
    ],
    "raw_predicates": [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://schema.org/codeRepository",
        "http://w3id.org/mlso/relatedTo",
    ],
}

# A record with no linked_entities at all
SPARSE_RECORD = {
    "model_id": "http://w3id.org/mlsea/pwc/model/TinyModel",
    "label": "TinyModel",
}

# A record with linked_entities + non-empty tasks (rare but should work)
RECORD_WITH_TASKS = {
    "model_id": "http://w3id.org/mlsea/pwc/model/GPT-2 Small",
    "label": "GPT-2 Small",
    "tasks": ["Language Modeling"],
    "linked_entities": [
        {
            "predicate": "http://schema.org/codeRepository",
            "predicate_label": "codeRepository",
            "object_uri": "https://github.com/openai/gpt-2",
            "object_label": "https://github.com/openai/gpt-2",
            "object_types": [],
        },
    ],
    "raw_predicates": ["http://www.w3.org/2000/01/rdf-schema#label"],
}


class TestModelGraphHelpers(unittest.TestCase):
    def test_extract_repo_urls(self) -> None:
        urls = extract_repo_urls(RICH_RECORD["linked_entities"])
        self.assertEqual(len(urls), 2)
        self.assertIn("https://github.com/pytorch/vision", urls)

    def test_extract_repo_names(self) -> None:
        names = extract_repo_names(RICH_RECORD["linked_entities"])
        self.assertIn("pytorch/vision", names)
        self.assertIn("keras-team/keras-applications", names)

    def test_extract_neighbor_labels(self) -> None:
        labels = extract_neighbor_labels(RICH_RECORD["linked_entities"])
        self.assertIn("Residual Connections", labels)
        # repo entities are excluded from neighbor labels
        self.assertNotIn("https://github.com/pytorch/vision", labels)

    def test_empty_linked_entities(self) -> None:
        self.assertEqual(extract_repo_urls([]), [])
        self.assertEqual(extract_repo_names([]), [])
        self.assertEqual(extract_neighbor_labels([]), [])

    def test_dedup_repo_urls(self) -> None:
        dup_entities = [
            {"predicate_label": "codeRepository", "object_uri": "https://github.com/a/b", "object_label": "https://github.com/a/b"},
            {"predicate_label": "codeRepository", "object_uri": "https://github.com/a/b", "object_label": "https://github.com/a/b"},
        ]
        urls = extract_repo_urls(dup_entities)
        self.assertEqual(len(urls), 1)


class TestModelTitleOnly(unittest.TestCase):
    def test_returns_prefixed_label(self) -> None:
        text = build_model_title_only_text(RICH_RECORD, {"max_characters": 512})
        self.assertEqual(text, "Model: ResNet-50")

    def test_empty_record(self) -> None:
        text = build_model_title_only_text({}, {"max_characters": 512})
        self.assertEqual(text, "")


class TestModelMetadata(unittest.TestCase):
    def test_rich_record_includes_repos_and_neighbors(self) -> None:
        text = build_model_metadata_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: ResNet-50", text)
        self.assertIn("Repositories:", text)
        self.assertIn("pytorch/vision", text)
        self.assertIn("Linked Entities:", text)
        self.assertIn("Residual Connections", text)

    def test_sparse_record(self) -> None:
        text = build_model_metadata_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: TinyModel", text)
        self.assertNotIn("Repositories:", text)

    def test_tasks_included_when_present(self) -> None:
        text = build_model_metadata_text(RECORD_WITH_TASKS, DEFAULT_CONFIG)
        self.assertIn("Tasks:", text)
        self.assertIn("Language Modeling", text)


class TestModelPredicateFiltered(unittest.TestCase):
    def test_rich_record_kept(self) -> None:
        text = build_model_predicate_filtered_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIsNotNone(text)
        self.assertIn("Model: ResNet-50", text)
        self.assertIn("Repositories:", text)

    def test_sparse_record_filtered_out(self) -> None:
        text = build_model_predicate_filtered_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIsNone(text)

    def test_record_with_linked_entities_kept(self) -> None:
        record_with_entities = {
            "model_id": "http://w3id.org/mlsea/pwc/model/X",
            "label": "X",
            "linked_entities": [
                {"predicate_label": "codeRepository", "object_uri": "https://github.com/x/y", "object_label": "https://github.com/x/y"},
            ],
        }
        text = build_model_predicate_filtered_text(record_with_entities, DEFAULT_CONFIG)
        self.assertIsNotNone(text)


class TestModelEnrichedMetadata(unittest.TestCase):
    def test_rich_record_includes_graph_sections(self) -> None:
        text = build_model_enriched_metadata_text(RICH_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: ResNet-50", text)
        self.assertIn("Model Family: ResNet", text)
        self.assertIn("Repositories:", text)
        self.assertIn("pytorch/vision", text)
        self.assertIn("Linked Entities:", text)
        self.assertIn("Residual Connections", text)
        self.assertIn("Predicates:", text)

    def test_sparse_record(self) -> None:
        text = build_model_enriched_metadata_text(SPARSE_RECORD, DEFAULT_CONFIG)
        self.assertIn("Model: TinyModel", text)
        self.assertNotIn("Repositories:", text)

    def test_empty_record(self) -> None:
        text = build_model_enriched_metadata_text({}, DEFAULT_CONFIG)
        self.assertEqual(text, "")

    def test_respects_max_characters(self) -> None:
        cfg = {**DEFAULT_CONFIG, "max_characters": 50}
        text = build_model_enriched_metadata_text(RICH_RECORD, cfg)
        self.assertLessEqual(len(text), 60)  # small overflow for truncation marker

    def test_model_family_extraction(self) -> None:
        text = build_model_enriched_metadata_text(RECORD_WITH_TASKS, DEFAULT_CONFIG)
        self.assertIn("Model Family: GPT-2", text)

    def test_tasks_included_when_present(self) -> None:
        text = build_model_enriched_metadata_text(RECORD_WITH_TASKS, DEFAULT_CONFIG)
        self.assertIn("Tasks:", text)
        self.assertIn("Language Modeling", text)


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
