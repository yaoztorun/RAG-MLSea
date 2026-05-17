"""
validate_question_dataset.py
Validate the ML questions dataset against schema requirements and canonical records.

Usage:
    python scripts/validate_question_dataset.py data/questions/ml_questions_dataset.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPERS_MASTER   = REPO_ROOT / "data/intermediate/raw_papers/papers_master.jsonl"
DATASETS_MASTER = REPO_ROOT / "data/intermediate/raw_datasets/datasets_master.jsonl"
MODELS_MASTER   = REPO_ROOT / "data/intermediate/raw_models/models_master.jsonl"

VALID_CATEGORIES   = {"paper", "dataset", "model"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_ANSWER_TYPES = {"literal", "entity", "unanswerable"}

PAPER_TYPES = {
    "paper_title_to_entity", "paper_description_to_title", "paper_task_to_paper",
    "paper_dataset_to_paper", "paper_method_to_paper", "paper_metric_to_paper",
    "paper_author_to_paper", "paper_repository_to_paper", "paper_abstract_semantic",
    "paper_multi_metadata",
}
DATASET_TYPES = {
    "dataset_title_to_entity", "dataset_description_to_title", "dataset_task_to_dataset",
    "dataset_keyword_to_dataset", "dataset_paper_to_dataset", "dataset_repository_to_dataset",
    "dataset_multi_metadata",
}
MODEL_TYPES = {
    "model_name_to_entity", "model_description_to_model", "model_family_variant",
    "model_task_to_model", "model_dataset_to_model", "model_paper_to_model",
    "model_repository_to_model", "model_multi_metadata",
}
UNANSWERABLE_TYPES = {
    "unanswerable_paper_title", "unanswerable_paper_metadata",
    "unanswerable_dataset_title", "unanswerable_dataset_metadata",
    "unanswerable_model_name", "unanswerable_model_metadata",
    "unanswerable_cross_entity_relation",
}

OLD_PROPERTY_TYPES = {
    "paper_to_authors", "paper_to_tasks", "paper_to_task_count",
    "paper_to_author_count", "paper_to_publication_year", "paper_to_keywords",
    "dataset_to_tasks", "dataset_to_task_count", "dataset_to_publication_year",
    "dataset_to_task_membership", "paper_to_implementation",
}

REQUIRED_FIELDS = {
    "id", "question", "question_type", "category", "difficulty",
    "target_entity_iri", "answer_type", "is_answerable", "source_fields",
}


def normalize_iri(iri: str) -> str:
    if not iri:
        return ""
    s = iri.strip()
    prev = s
    cur = unquote(prev)
    while cur != prev:
        prev = cur
        cur = unquote(prev)
    return cur.strip()


def category_from_iri(iri: str) -> str:
    n = normalize_iri(str(iri))
    if "/scientificWork/" in n:
        return "paper"
    if "/dataset/" in n:
        return "dataset"
    if "/model/" in n:
        return "model"
    return "unknown"


def _load_paper_id_set() -> set[str]:
    print("[Canon] Loading paper IDs ...", flush=True)
    ids: set = set()
    with PAPERS_MASTER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = normalize_iri(str(r.get("paper_id", "")))
            if pid:
                ids.add(pid)
    print(f"[Canon] {len(ids)} paper IDs loaded", flush=True)
    return ids


def _load_dataset_id_set() -> tuple[set[str], dict[str, dict]]:
    print("[Canon] Loading dataset IDs ...", flush=True)
    ids: set = set()
    records: dict = {}
    with DATASETS_MASTER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = normalize_iri(str(r.get("dataset_id", "")))
            if did:
                ids.add(did)
                records[did] = r
    print(f"[Canon] {len(ids)} dataset IDs loaded", flush=True)
    return ids, records


def _load_model_id_set() -> tuple[set[str], dict[str, dict]]:
    print("[Canon] Loading model IDs ...", flush=True)
    ids: set = set()
    records: dict = {}
    with MODELS_MASTER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mid = normalize_iri(str(r.get("model_id", "")))
            if mid:
                ids.add(mid)
                records[mid] = r
    print(f"[Canon] {len(ids)} model IDs loaded", flush=True)
    return ids, records


def _pick_example(questions: list[dict], category: str) -> dict | None:
    for q in questions:
        if q.get("is_answerable", True) and q.get("category") == category and q.get("target_entity_iri"):
            return q
    return None


def validate(dataset_path: Path) -> int:
    print(f"\n=== Validating {dataset_path.name} ===\n", flush=True)

    with dataset_path.open(encoding="utf-8") as fh:
        questions: list[dict] = json.load(fh)

    total = len(questions)
    answerable   = [q for q in questions if q.get("is_answerable", True)]
    unanswerable = [q for q in questions if not q.get("is_answerable", True)]

    # ── Load canonical indexes ────────────────────────────────────────────────
    paper_ids = _load_paper_id_set()
    dataset_ids, dataset_records = _load_dataset_id_set()
    model_ids, model_records = _load_model_id_set()

    print()

    # ── Basic counts ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("BASIC COUNTS")
    print("=" * 60)
    print(f"  Total questions   : {total}")
    print(f"  Answerable        : {len(answerable)}")
    print(f"  Unanswerable      : {len(unanswerable)}")
    print()

    # ── Distributions ─────────────────────────────────────────────────────────
    cat_ctr  = Counter(q.get("category") for q in answerable)
    diff_ctr = Counter(q.get("difficulty") for q in answerable)
    qt_ctr   = Counter(q.get("question_type") for q in questions)

    print("CATEGORY DISTRIBUTION (answerable)")
    for cat, n in sorted(cat_ctr.items()):
        print(f"  {cat:<20}: {n}")
    print()

    print("DIFFICULTY DISTRIBUTION (answerable)")
    for diff, n in sorted(diff_ctr.items()):
        print(f"  {diff:<20}: {n}")
    print()

    print("QUESTION TYPE DISTRIBUTION")
    for qt, n in qt_ctr.most_common():
        print(f"  {qt:<45}: {n}")
    print()

    # ── IRI diversity ─────────────────────────────────────────────────────────
    iris = [normalize_iri(str(q["target_entity_iri"])) for q in answerable if q.get("target_entity_iri")]
    iri_ctr = Counter(iris)
    unique_iris = len(iri_ctr)
    max_iri = max(iri_ctr.values()) if iri_ctr else 0
    avg_iri = len(iris) / unique_iris if unique_iris else 0

    iri_by_cat: dict[str, set] = defaultdict(set)
    for q in answerable:
        c = q.get("category"); iri = q.get("target_entity_iri")
        if c and iri:
            iri_by_cat[c].add(normalize_iri(str(iri)))

    print("GOLD TARGET IRI DIVERSITY")
    print(f"  Unique IRIs (total answerable) : {unique_iris}")
    for cat, s in sorted(iri_by_cat.items()):
        print(f"  Unique {cat} IRIs              : {len(s)}")
    print(f"  Average questions per IRI      : {avg_iri:.2f}")
    print(f"  Maximum questions per IRI      : {max_iri}")
    print()

    # ── Hard error tracking ───────────────────────────────────────────────────
    hard_errors: list[str] = []
    warnings: list[str] = []

    # ── Duplicate IDs ──────────────────────────────────────────────────────────
    id_ctr = Counter(q.get("id") for q in questions)
    dup_ids = [qid for qid, n in id_ctr.items() if n > 1]
    if dup_ids:
        for qid in dup_ids[:10]:
            hard_errors.append(f"Duplicate question ID: {qid}")
    print(f"DUPLICATE IDs: {len(dup_ids)} found")
    if dup_ids:
        for qid in dup_ids[:5]:
            print(f"  !! {qid}")
    print()

    # ── Duplicate question texts ───────────────────────────────────────────────
    text_ctr = Counter(q.get("question", "").strip() for q in questions)
    dup_texts = [(t, n) for t, n in text_ctr.items() if n > 1 and t]
    print(f"DUPLICATE QUESTION TEXTS: {len(dup_texts)} found")
    for t, n in dup_texts[:5]:
        warnings.append(f"Duplicate question text (×{n}): {t[:80]}")
        print(f"  [x{n}] {t[:80]}")
    print()

    # ── Missing required fields ────────────────────────────────────────────────
    missing_fields: dict[str, list] = defaultdict(list)
    for q in questions:
        qid = q.get("id", "?")
        for f in REQUIRED_FIELDS:
            if f not in q or q[f] is None and f not in {"target_entity_iri"}:
                missing_fields[f].append(qid)
        # source_fields specifically must be a list
        if not isinstance(q.get("source_fields"), list):
            missing_fields["source_fields (not a list)"].append(qid)

    print("MISSING REQUIRED FIELDS")
    any_missing = False
    for field, qids in sorted(missing_fields.items()):
        n = len(qids)
        if n > 0:
            any_missing = True
            examples = qids[:3]
            print(f"  {field:<40}: {n} questions (e.g. {examples})")
            if field in {"id", "question", "question_type", "category"}:
                hard_errors.append(f"Missing field '{field}' in {n} questions")
    if not any_missing:
        print("  All required fields present.")
    print()

    # ── Invalid field values ───────────────────────────────────────────────────
    invalid_diff  = [q for q in answerable if q.get("difficulty") not in VALID_DIFFICULTIES]
    invalid_cat   = [q for q in answerable if q.get("category") not in VALID_CATEGORIES]
    invalid_atype = [q for q in questions if q.get("answer_type") not in VALID_ANSWER_TYPES and q.get("answer_type") is not None]

    print("INVALID FIELD VALUES")
    print(f"  Invalid difficulty (answerable) : {len(invalid_diff)}")
    if invalid_diff:
        for q in invalid_diff[:3]:
            hard_errors.append(f"Invalid difficulty '{q.get('difficulty')}' on {q.get('id')}")
            print(f"    !! {q.get('id')}: difficulty={q.get('difficulty')}")
    print(f"  Invalid category (answerable)   : {len(invalid_cat)}")
    if invalid_cat:
        for q in invalid_cat[:3]:
            hard_errors.append(f"Invalid category '{q.get('category')}' on {q.get('id')}")
            print(f"    !! {q.get('id')}: category={q.get('category')}")
    print(f"  Invalid answer_type             : {len(invalid_atype)}")
    if invalid_atype:
        for q in invalid_atype[:3]:
            warnings.append(f"Invalid answer_type '{q.get('answer_type')}' on {q.get('id')}")
            print(f"    [w] {q.get('id')}: answer_type={q.get('answer_type')}")
    print()

    # ── Answerable with null/empty IRI ────────────────────────────────────────
    ans_no_iri = [q for q in answerable if not q.get("target_entity_iri")]
    print(f"ANSWERABLE WITH NULL/EMPTY target_entity_iri: {len(ans_no_iri)}")
    for q in ans_no_iri[:5]:
        hard_errors.append(f"Answerable question {q.get('id')} has no target_entity_iri")
        print(f"  !! {q.get('id')}")
    print()

    # ── Answerable with empty source_fields ───────────────────────────────────
    ans_no_sf = [q for q in answerable if not q.get("source_fields")]
    print(f"ANSWERABLE WITH EMPTY source_fields: {len(ans_no_sf)}")
    for q in ans_no_sf[:5]:
        warnings.append(f"Answerable question {q.get('id')} has empty source_fields")
        print(f"  [w] {q.get('id')}: {q.get('question', '')[:60]}")
    print()

    # ── Unanswerable with non-null IRI ────────────────────────────────────────
    unans_with_iri = [q for q in unanswerable if q.get("target_entity_iri")]
    print(f"UNANSWERABLE WITH NON-NULL target_entity_iri: {len(unans_with_iri)}")
    for q in unans_with_iri[:5]:
        warnings.append(f"Unanswerable question {q.get('id')} has a target_entity_iri")
        print(f"  [w] {q.get('id')}")
    print()

    # ── Old property-extraction types ─────────────────────────────────────────
    old_type_qs = [q for q in questions if q.get("question_type") in OLD_PROPERTY_TYPES]
    print(f"OLD PROPERTY-EXTRACTION QUESTION TYPES REMAINING: {len(old_type_qs)}")
    for q in old_type_qs[:5]:
        hard_errors.append(f"Old property type '{q.get('question_type')}' still present on {q.get('id')}")
        print(f"  !! {q.get('id')}: type={q.get('question_type')}")
    print()

    # ── IRI existence in canonical records ────────────────────────────────────
    print("IRI EXISTENCE IN CANONICAL RECORDS")
    missing_iris: list[dict] = []
    for q in answerable:
        iri_raw = normalize_iri(str(q.get("target_entity_iri") or ""))
        cat = q.get("category", "")
        if not iri_raw:
            continue
        found = (
            (cat == "paper"   and iri_raw in paper_ids)   or
            (cat == "dataset" and iri_raw in dataset_ids) or
            (cat == "model"   and iri_raw in model_ids)
        )
        if not found:
            missing_iris.append({"id": q.get("id"), "category": cat, "iri": iri_raw})
    print(f"  IRIs not found in canonical records: {len(missing_iris)}")
    for entry in missing_iris[:10]:
        hard_errors.append(f"IRI not in canonical: {entry['id']} ({entry['iri'][:60]})")
        print(f"  !! {entry['id']} ({entry['category']}): {entry['iri'][:80]}")
    print()

    # ── Category / IRI namespace mismatch ─────────────────────────────────────
    print("CATEGORY / IRI NAMESPACE MISMATCHES")
    mismatch = []
    for q in answerable:
        iri = str(q.get("target_entity_iri") or "")
        if not iri:
            continue
        iri_cat = category_from_iri(iri)
        cat = q.get("category", "")
        if iri_cat != "unknown" and iri_cat != cat:
            mismatch.append({"id": q.get("id"), "category": cat, "iri_category": iri_cat, "iri": iri[:60]})
    print(f"  Mismatches: {len(mismatch)}")
    for m in mismatch[:5]:
        hard_errors.append(f"Category mismatch on {m['id']}: category={m['category']} but IRI implies {m['iri_category']}")
        print(f"  !! {m['id']}: category={m['category']} but IRI implies {m['iri_category']}")
    print()

    # ── Question type / category mismatch ─────────────────────────────────────
    print("QUESTION TYPE / CATEGORY MISMATCHES")
    qt_cat_mismatch = []
    for q in answerable:
        qt  = q.get("question_type", "")
        cat = q.get("category", "")
        if qt in PAPER_TYPES and cat != "paper":
            qt_cat_mismatch.append((q.get("id"), qt, cat))
        elif qt in DATASET_TYPES and cat != "dataset":
            qt_cat_mismatch.append((q.get("id"), qt, cat))
        elif qt in MODEL_TYPES and cat != "model":
            qt_cat_mismatch.append((q.get("id"), qt, cat))
    print(f"  Mismatches: {len(qt_cat_mismatch)}")
    for qid, qt, cat in qt_cat_mismatch[:5]:
        warnings.append(f"QType/category mismatch {qid}: qt={qt} but category={cat}")
        print(f"  [w] {qid}: qt={qt}, category={cat}")
    print()

    # ── IRI normalization diagnostic ──────────────────────────────────────────
    print("=" * 60)
    print("IRI MATCH DIAGNOSTIC (one example per category)")
    print("=" * 60)
    for cat in ["paper", "dataset", "model"]:
        example_q = _pick_example(questions, cat)
        if example_q is None:
            print(f"\n  [{cat.upper()} example] No answerable {cat} question found.\n")
            continue
        q_iri = str(example_q.get("target_entity_iri", ""))
        norm_iri = normalize_iri(q_iri)

        canonical_id = None
        if cat == "paper" and norm_iri in paper_ids:
            canonical_id = norm_iri
        elif cat == "dataset" and norm_iri in dataset_ids:
            canonical_id = norm_iri
        elif cat == "model" and norm_iri in model_ids:
            canonical_id = norm_iri

        match = "YES" if canonical_id else "NO"
        print(f"\n  [{cat.upper()} example]")
        print(f"    question_id              : {example_q.get('id')}")
        print(f"    question target_entity_iri: {q_iri[:100]}")
        print(f"    normalize_iri(gold)       : {norm_iri[:100]}")
        print(f"    canonical record id found : {canonical_id[:100] if canonical_id else 'NOT FOUND'}")
        print(f"    expected retrieval cand ID: {norm_iri[:100]}")
        print(f"    match                     : {match}")
        if match == "NO":
            hard_errors.append(f"IRI normalization diagnostic failed for {cat} example {example_q.get('id')}")

    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Hard errors : {len(hard_errors)}")
    for e in hard_errors[:20]:
        print(f"    !! {e}")
    print(f"  Warnings    : {len(warnings)}")
    for w in warnings[:10]:
        print(f"    [w] {w}")
    print()

    if hard_errors:
        print("RESULT: FAIL (hard errors present)")
        return 1

    print("RESULT: PASS")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dataset_path>", file=sys.stderr)
        return 2
    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"File not found: {dataset_path}", file=sys.stderr)
        return 2
    return validate(dataset_path)


if __name__ == "__main__":
    raise SystemExit(main())
