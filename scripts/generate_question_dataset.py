"""
generate_question_dataset.py
Expand and clean data/questions/ml_questions_dataset.json in-place.
Target: 520 questions (500 answerable, 20 unanswerable).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH   = REPO_ROOT / "data/questions/ml_questions_dataset.json"
BACKUP_PATH      = REPO_ROOT / "data/questions/ml_questions_dataset_backup_before_expansion.json"
PAPERS_MASTER    = REPO_ROOT / "data/intermediate/raw_papers/papers_master.jsonl"
DATASETS_MASTER  = REPO_ROOT / "data/intermediate/raw_datasets/datasets_master.jsonl"
MODELS_MASTER    = REPO_ROOT / "data/intermediate/raw_models/models_master.jsonl"
SUMMARY_PATH     = REPO_ROOT / "data/questions/ml_questions_dataset_summary.md"
AUDIT_PATH       = REPO_ROOT / "data/questions/ml_questions_dataset_audit.md"
MISSING_PATH     = REPO_ROOT / "data/questions/missing_gold_targets.json"

# ── Targets ──────────────────────────────────────────────────────────────────

TARGET_TOTAL       = 520
TARGET_ANSWERABLE  = 500
TARGET_UNANSWERABLE = 20
TARGET_PAPER       = 250
TARGET_DATASET     = 125
TARGET_MODEL       = 125
MAX_USES_PER_IRI   = 2

# ── Taxonomy ─────────────────────────────────────────────────────────────────

VALID_CATEGORIES   = {"paper", "dataset", "model"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

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

PROPERTY_TYPES = {
    "paper_to_authors", "paper_to_tasks", "paper_to_task_count",
    "paper_to_author_count", "paper_to_publication_year", "paper_to_keywords",
    "dataset_to_tasks", "dataset_to_task_count", "dataset_to_publication_year",
    "dataset_to_task_membership",
}

# Old → new question_type
QTYPE_REMAP = {
    "paper_by_author_and_task":         "paper_multi_metadata",
    "paper_by_task_pair":               "paper_multi_metadata",
    "paper_to_implementation":          "paper_repository_to_paper",  # will be further REPLACEd
    "title_and_year_filter_to_papers":  "paper_multi_metadata",
    "semantic_task_to_dataset":         "dataset_task_to_dataset",
    "task_to_dataset":                  "dataset_task_to_dataset",
    "tasks_to_dataset":                 "dataset_multi_metadata",
    "repository_to_model":              "model_repository_to_model",
    "semantic_repository_to_model":     "model_repository_to_model",
    "comparative_model_variant":        "model_family_variant",
    "model_name_resolution":            "model_name_to_entity",
}

# Old → new category
CATEGORY_REMAP = {
    "semantic":      "paper",
    "multihop":      "paper",
    "implementation":"paper",
}

DIFFICULTY_DEFAULT = {
    "paper_title_to_entity": "easy",
    "model_name_to_entity":  "easy",
    "dataset_title_to_entity": "easy",
    "paper_author_to_paper": "medium",
    "paper_task_to_paper":   "medium",
    "paper_repository_to_paper": "medium",
    "paper_dataset_to_paper": "medium",
    "paper_method_to_paper": "medium",
    "paper_metric_to_paper": "medium",
    "dataset_task_to_dataset": "medium",
    "dataset_keyword_to_dataset": "medium",
    "dataset_paper_to_dataset": "medium",
    "model_repository_to_model": "medium",
    "model_paper_to_model": "medium",
    "paper_description_to_title": "hard",
    "paper_abstract_semantic": "hard",
    "paper_multi_metadata": "hard",
    "dataset_multi_metadata": "hard",
    "model_family_variant": "hard",
    "model_multi_metadata": "hard",
}

MODEL_PREFIX   = "http://w3id.org/mlsea/pwc/model/"
PAPER_PREFIX   = "http://w3id.org/mlsea/pwc/scientificWork/"
DATASET_PREFIX = "http://w3id.org/mlsea/pwc/dataset/"

# ── IRI utilities ────────────────────────────────────────────────────────────

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
    return ""


def _tmpl(templates: list[str], key: str) -> str:
    """Pick template deterministically by hashing key."""
    return templates[abs(hash(key)) % len(templates)]


def _abbrev(s: str, maxlen: int = 60) -> str:
    s = str(s or "").strip()
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen].rsplit(" ", 1)
    return (cut[0] if len(cut) > 1 else s[:maxlen]) + "…"


def _abstract_fragment(abstract: str, title: str, max_chars: int = 150) -> str:
    """Extract a short, title-free abstract fragment."""
    text = abstract.strip()
    # Find first sentence boundary within max_chars * 2
    for end in range(min(len(text), max_chars * 2), 0, -1):
        if text[end - 1] == "." and (end == len(text) or text[end] == " "):
            snippet = text[:end].strip()
            if len(snippet) <= max_chars:
                break
    else:
        snippet = text[:max_chars].rsplit(" ", 1)[0]
    # Check for title leakage (case-insensitive)
    t_lower = title.lower()[:40]
    if t_lower and t_lower in snippet.lower():
        # Try to use a later fragment
        alt = text[80:80 + max_chars].rsplit(" ", 1)[0].strip()
        if alt and t_lower not in alt.lower():
            snippet = alt
    return snippet if snippet else text[:max_chars].rsplit(" ", 1)[0]


# ── Canonical loading ─────────────────────────────────────────────────────────

def load_paper_index(used_iris: set, max_candidates: int = 3000) -> tuple[set, list]:
    """Stream papers_master; return (paper_id_set, candidate_list).
    Candidates are compact dicts for papers not in used_iris that have enough metadata.
    """
    print(f"[Canon] Loading papers from {PAPERS_MASTER.name} ...", flush=True)
    paper_id_set: set = set()
    candidates: list = []
    total = 0
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
            if not pid:
                continue
            paper_id_set.add(pid)
            total += 1
            if total % 50000 == 0:
                print(f"  … {total} paper IDs scanned", flush=True)
            if pid in used_iris:
                continue
            if len(candidates) >= max_candidates:
                continue
            abstract = str(r.get("abstract") or "").strip()
            authors = [a for a in (r.get("authors") or []) if a]
            tasks   = [t for t in (r.get("tasks")   or []) if t]
            if not abstract and not authors and not tasks:
                continue
            candidates.append({
                "paper_id": pid,
                "title":    str(r.get("title") or "").strip(),
                "abstract": abstract[:400],
                "authors":  authors[:5],
                "tasks":    tasks[:5],
            })
    print(f"[Canon] {total} paper IDs, {len(candidates)} candidates", flush=True)
    return paper_id_set, candidates


def load_dataset_index() -> dict:
    print(f"[Canon] Loading datasets ...", flush=True)
    idx: dict = {}
    with DATASETS_MASTER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = normalize_iri(str(r.get("dataset_id", "")))
            if did:
                idx[did] = r
    print(f"[Canon] {len(idx)} datasets loaded", flush=True)
    return idx


def load_model_index() -> tuple[dict, dict]:
    """Return (model_index, sibling_index).
    model_index: {model_id: record}
    sibling_index: {parent_paper_title: [compact_model]}
    """
    print(f"[Canon] Loading models ...", flush=True)
    model_idx: dict = {}
    sibling_idx: dict[str, list] = defaultdict(list)
    with MODELS_MASTER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mid = normalize_iri(str(r.get("model_id", "")))
            if not mid:
                continue
            model_idx[mid] = r
            # Parse parent paper title from model_id
            rest = mid[len(MODEL_PREFIX):]
            if "/" in rest:
                parent = rest.rsplit("/", 1)[0]
            else:
                parent = ""
            repos = list(dict.fromkeys(
                le["object_label"]
                for le in (r.get("linked_entities") or [])
                if le.get("predicate_label") == "codeRepository" and le.get("object_label")
            ))
            sibling_idx[parent].append({
                "model_id":   mid,
                "label":      str(r.get("label") or "").strip(),
                "parent":     parent,
                "repos":      repos,
            })
    print(f"[Canon] {len(model_idx)} models, {len(sibling_idx)} parent papers", flush=True)
    return model_idx, sibling_idx


# ── Audit helpers ─────────────────────────────────────────────────────────────

def _is_entity_retrieval(q: dict) -> bool:
    """Return True if question is a valid entity-retrieval question."""
    qt = q.get("question_type", "")
    cat = q.get("category", "")
    iri = q.get("target_entity_iri")
    if not q.get("is_answerable", True):
        return False  # handled separately
    if qt in PROPERTY_TYPES:
        return False
    # title_and_year_filter: directly names paper but still entity-retrieval
    all_known = PAPER_TYPES | DATASET_TYPES | MODEL_TYPES | set(QTYPE_REMAP.keys()) | {
        "paper_description_to_title", "dataset_to_task_membership",
    }
    if qt in PROPERTY_TYPES:
        return False
    if not iri:
        return False
    return True


def _infer_difficulty(qt: str) -> str:
    return DIFFICULTY_DEFAULT.get(qt, "medium")


def _infer_category(q: dict) -> str:
    cat = q.get("category", "")
    cat = CATEGORY_REMAP.get(cat, cat)
    if cat in VALID_CATEGORIES:
        return cat
    # Infer from IRI
    iri = q.get("target_entity_iri")
    if iri:
        c = category_from_iri(str(iri))
        if c:
            return c
    return ""


def _repair(q: dict, paper_id_set: set, dataset_idx: dict, model_idx: dict,
            iri_uses: Counter) -> tuple[dict | None, str]:
    """
    Returns (repaired_question_or_None, audit_note).
    Applies remapping, fills missing fields, validates IRI.
    """
    q = dict(q)
    qt_orig = q.get("question_type", "")
    iri = normalize_iri(str(q.get("target_entity_iri") or ""))

    # Remap question_type
    qt = QTYPE_REMAP.get(qt_orig, qt_orig)
    q["question_type"] = qt

    # Infer category
    cat = _infer_category(q)
    if not cat:
        return None, "discard: cannot determine category"
    q["category"] = cat
    q["target_entity_type"] = cat

    # Validate IRI is in canonical records
    if not iri:
        return None, "discard: no target_entity_iri"
    ok = False
    if cat == "paper" and iri in paper_id_set:
        ok = True
    elif cat == "dataset" and iri in dataset_idx:
        ok = True
    elif cat == "model" and iri in model_idx:
        ok = True
    if not ok:
        # Try un-encoded version
        iri_decoded = unquote(iri)
        if cat == "paper" and iri_decoded in paper_id_set:
            iri = iri_decoded; ok = True
        elif cat == "dataset" and iri_decoded in dataset_idx:
            iri = iri_decoded; ok = True
        elif cat == "model" and iri_decoded in model_idx:
            iri = iri_decoded; ok = True
    if not ok:
        return None, f"discard: IRI not in canonical records ({iri[:80]})"
    q["target_entity_iri"] = iri

    # IRI usage limit
    if iri_uses[iri] >= MAX_USES_PER_IRI:
        return None, f"discard: IRI already used {iri_uses[iri]} times"

    # Fix difficulty
    if q.get("difficulty") not in VALID_DIFFICULTIES:
        q["difficulty"] = _infer_difficulty(qt)

    # Fix answer_type
    if q.get("answer_type") not in {"literal", "entity"}:
        q["answer_type"] = "literal"

    # Fix answer / text_answer for paper_to_implementation (which is still property-ish)
    # These will be flagged as REPLACE separately.

    # Add source_fields if missing
    if not q.get("source_fields"):
        q["source_fields"] = _infer_source_fields(qt)

    # Ensure is_answerable
    q["is_answerable"] = True

    # Ensure text_answer is present
    if not q.get("text_answer"):
        ans = q.get("answer", "")
        q["text_answer"] = f"The answer is {ans}." if ans else "See canonical record."

    return q, "repair"


def _infer_source_fields(qt: str) -> list[str]:
    mapping = {
        "paper_title_to_entity":       ["title"],
        "paper_description_to_title":  ["existing_question_dataset"],
        "paper_multi_metadata":        ["existing_question_dataset"],
        "paper_author_to_paper":       ["authors", "tasks"],
        "paper_task_to_paper":         ["tasks"],
        "paper_repository_to_paper":   ["existing_question_dataset"],
        "paper_abstract_semantic":     ["abstract"],
        "dataset_task_to_dataset":     ["tasks"],
        "dataset_multi_metadata":      ["tasks", "keywords"],
        "dataset_title_to_entity":     ["title"],
        "dataset_keyword_to_dataset":  ["keywords"],
        "model_name_to_entity":        ["label"],
        "model_repository_to_model":   ["linked_entities", "codeRepository"],
        "model_family_variant":        ["model_id", "label"],
        "model_paper_to_model":        ["model_id", "paper_title"],
        "model_multi_metadata":        ["model_id", "paper_title", "linked_entities"],
    }
    return mapping.get(qt, ["existing_question_dataset"])


# ── Replacement builders ───────────────────────────────────────────────────────

def _build_replacement_paper(
    old_q: dict, paper_id: str, paper_id_set: set,
    paper_candidates: list, iri_uses: Counter
) -> dict | None:
    """Try to create an entity-retrieval question for an existing paper IRI."""
    # Find the canonical record in candidates
    rec = next((c for c in paper_candidates if c["paper_id"] == paper_id), None)
    if rec is None:
        return None  # Can't generate without record
    authors = rec["authors"]
    tasks   = rec["tasks"]
    abstract = rec["abstract"]
    title    = rec["title"]

    if authors and tasks:
        # paper_author_to_paper (medium)
        a = authors[0]
        t = tasks[0]
        tmpls = [
            f"Which paper by {a} addresses the task of {t}?",
            f"What paper authored by {a} focuses on {t}?",
            f"Which research paper by {a} covers {t}?",
        ]
        q_text = _tmpl(tmpls, paper_id + "rep")
        qt = "paper_author_to_paper"
        diff = "medium"
        sf = ["authors", "tasks"]
    elif abstract:
        # paper_abstract_semantic (hard)
        frag = _abstract_fragment(abstract, title)
        tmpls = [
            f"Which paper begins its abstract with: '{frag}'?",
            f"Which paper contains this excerpt in its abstract: '{frag}'?",
        ]
        q_text = _tmpl(tmpls, paper_id + "rep")
        qt = "paper_abstract_semantic"
        diff = "hard"
        sf = ["abstract"]
    else:
        # paper_title_to_entity (easy)
        tmpls = [
            f"Which paper has the title '{title}'?",
            f"What paper is titled '{title}'?",
        ]
        q_text = _tmpl(tmpls, paper_id + "rep")
        qt = "paper_title_to_entity"
        diff = "easy"
        sf = ["title"]

    return {
        "question":           q_text,
        "question_type":      qt,
        "category":           "paper",
        "target_entity_type": "paper",
        "difficulty":         diff,
        "target_entity_iri":  paper_id,
        "answer":             title,
        "answer_type":        "literal",
        "text_answer":        f"The paper is '{title}'.",
        "is_answerable":      True,
        "source_fields":      sf,
    }


def _build_replacement_dataset(
    old_q: dict, dataset_id: str, dataset_idx: dict, iri_uses: Counter
) -> dict | None:
    rec = dataset_idx.get(dataset_id)
    if not rec:
        return None
    title    = str(rec.get("title") or rec.get("label") or "").strip()
    tasks    = [t for t in (rec.get("tasks") or []) if t]
    keywords = [k for k in (rec.get("keywords") or []) if k]

    if tasks and keywords:
        t = tasks[0]; k = keywords[0]
        tmpls = [
            f"Which dataset associated with {t} also has the keyword {k}?",
            f"What dataset is used for {t} and is tagged with {k}?",
        ]
        q_text = _tmpl(tmpls, dataset_id + "rep")
        qt = "dataset_task_to_dataset"
        diff = "medium"
        sf = ["tasks", "keywords"]
    elif tasks:
        t = tasks[0]
        tmpls = [
            f"Which dataset is used for {t} tasks?",
            f"What dataset supports the {t} task?",
        ]
        q_text = _tmpl(tmpls, dataset_id + "rep")
        qt = "dataset_task_to_dataset"
        diff = "medium"
        sf = ["tasks"]
    elif keywords:
        k = keywords[0]
        tmpls = [
            f"Which dataset is associated with the keyword {k}?",
            f"What dataset is tagged with {k}?",
        ]
        q_text = _tmpl(tmpls, dataset_id + "rep")
        qt = "dataset_keyword_to_dataset"
        diff = "medium"
        sf = ["keywords"]
    else:
        tmpls = [
            f"Which dataset is named '{title}'?",
            f"What dataset has the title '{title}'?",
        ]
        q_text = _tmpl(tmpls, dataset_id + "rep")
        qt = "dataset_title_to_entity"
        diff = "easy"
        sf = ["title"]

    return {
        "question":           q_text,
        "question_type":      qt,
        "category":           "dataset",
        "target_entity_type": "dataset",
        "difficulty":         diff,
        "target_entity_iri":  dataset_id,
        "answer":             title,
        "answer_type":        "literal",
        "text_answer":        f"The dataset is '{title}'.",
        "is_answerable":      True,
        "source_fields":      sf,
    }


# ── Question generators ────────────────────────────────────────────────────────

def generate_paper_questions(
    paper_candidates: list, iri_uses: Counter, target: int,
    dataset_clue_idx: dict[str, set], used_texts: set
) -> list[dict]:
    """Generate paper-category questions. dataset_clue_idx maps task->set of paper_ids for ambiguity check."""
    out: list[dict] = []
    need = target

    # Group candidates by what's available
    with_authors_tasks    = [c for c in paper_candidates if c["authors"] and c["tasks"]]
    with_two_tasks        = [c for c in paper_candidates if len(c["tasks"]) >= 2]
    with_abstract         = [c for c in paper_candidates if c["abstract"]]
    with_authors_abstract = [c for c in paper_candidates if c["authors"] and c["abstract"]]
    all_cands             = paper_candidates

    # Build author+task clue index for ambiguity check
    author_task_to_ids: dict[tuple, set] = defaultdict(set)
    for c in paper_candidates:
        for a in c["authors"][:2]:
            for t in c["tasks"][:2]:
                author_task_to_ids[(a, t)].add(c["paper_id"])

    # Build task+task clue index
    task_pair_to_ids: dict[tuple, set] = defaultdict(set)
    for c in paper_candidates:
        tasks = c["tasks"]
        if len(tasks) >= 2:
            task_pair_to_ids[(tasks[0], tasks[1])].add(c["paper_id"])

    def _try_add(q_dict: dict) -> bool:
        iri = q_dict["target_entity_iri"]
        if iri_uses[iri] >= MAX_USES_PER_IRI:
            return False
        text = q_dict["question"].strip()
        if text in used_texts:
            return False
        out.append(q_dict)
        iri_uses[iri] += 1
        used_texts.add(text)
        return True

    # --- Easy: paper_title_to_entity ---
    easy_target = math.ceil(need * 0.28)  # ~28% easy
    title_tmpls = [
        "Which paper has the title '{title}'?",
        "What paper is titled '{title}'?",
        "Find the paper entitled '{title}'.",
        "Which MLSea paper corresponds to the title '{title}'?",
    ]
    added_easy = 0
    for c in all_cands:
        if added_easy >= easy_target:
            break
        pid = c["paper_id"]; title = c["title"]
        if not title or iri_uses[pid] >= MAX_USES_PER_IRI:
            continue
        tmpl = _tmpl(title_tmpls, pid + "e")
        q_text = tmpl.format(title=_abbrev(title, 120))
        _try_add({
            "question": q_text, "question_type": "paper_title_to_entity",
            "category": "paper", "target_entity_type": "paper", "difficulty": "easy",
            "target_entity_iri": pid, "answer": title, "answer_type": "literal",
            "text_answer": f"The paper is '{_abbrev(title, 120)}'.",
            "is_answerable": True, "source_fields": ["title"],
        })
        added_easy += 1

    # --- Medium: paper_author_to_paper ---
    medium_target = math.ceil(need * 0.36)  # ~36% medium
    author_tmpls = [
        "Which paper by {author} addresses the task of {task}?",
        "What paper authored by {author} focuses on {task}?",
        "Which research paper by {author} covers {task}?",
        "What work by {author} is specifically about {task}?",
    ]
    added_med = 0
    for c in with_authors_tasks:
        if added_med >= medium_target // 2:
            break
        pid = c["paper_id"]
        if iri_uses[pid] >= MAX_USES_PER_IRI:
            continue
        a = c["authors"][0]; t = c["tasks"][0]
        # Ambiguity check: need unique author+task combo
        if len(author_task_to_ids.get((a, t), set())) > 1:
            # Try second task or second author
            found = False
            for a2 in c["authors"][:3]:
                for t2 in c["tasks"][:3]:
                    if len(author_task_to_ids.get((a2, t2), set())) == 1:
                        a, t = a2, t2; found = True; break
                if found:
                    break
            if not found:
                continue  # Skip ambiguous
        tmpl = _tmpl(author_tmpls, pid + "m1")
        q_text = tmpl.format(author=a, task=t)
        if _try_add({
            "question": q_text, "question_type": "paper_author_to_paper",
            "category": "paper", "target_entity_type": "paper", "difficulty": "medium",
            "target_entity_iri": pid,
            "answer": c["title"], "answer_type": "literal",
            "text_answer": f"The paper is '{_abbrev(c['title'], 100)}'.",
            "is_answerable": True, "source_fields": ["authors", "tasks"],
        }):
            added_med += 1

    # --- Medium: paper_task_to_paper (two-task) ---
    task_pair_tmpls = [
        "Which paper addresses both {task1} and {task2}?",
        "What paper focuses on {task1} as well as {task2}?",
        "Which research paper covers the tasks of {task1} and {task2}?",
        "What work is specifically about both {task1} and {task2}?",
    ]
    for c in with_two_tasks:
        if added_med >= medium_target:
            break
        pid = c["paper_id"]
        if iri_uses[pid] >= MAX_USES_PER_IRI:
            continue
        t1 = c["tasks"][0]; t2 = c["tasks"][1]
        # Ambiguity check
        if len(task_pair_to_ids.get((t1, t2), set())) > 1:
            continue
        tmpl = _tmpl(task_pair_tmpls, pid + "m2")
        q_text = tmpl.format(task1=t1, task2=t2)
        if _try_add({
            "question": q_text, "question_type": "paper_task_to_paper",
            "category": "paper", "target_entity_type": "paper", "difficulty": "medium",
            "target_entity_iri": pid,
            "answer": c["title"], "answer_type": "literal",
            "text_answer": f"The paper is '{_abbrev(c['title'], 100)}'.",
            "is_answerable": True, "source_fields": ["tasks"],
        }):
            added_med += 1

    # --- Hard: paper_abstract_semantic ---
    hard_target = need - easy_target - medium_target
    abstract_tmpls = [
        "Which paper begins its abstract with: '{frag}'?",
        "Which paper contains this excerpt in its abstract: '{frag}'?",
        "Which paper's abstract opens with the following: '{frag}'?",
    ]
    added_hard = 0
    for c in with_abstract:
        if added_hard >= hard_target // 2:
            break
        pid = c["paper_id"]
        if iri_uses[pid] >= MAX_USES_PER_IRI:
            continue
        frag = _abstract_fragment(c["abstract"], c["title"])
        if len(frag) < 30:
            continue
        tmpl = _tmpl(abstract_tmpls, pid + "h1")
        q_text = tmpl.format(frag=frag)
        if _try_add({
            "question": q_text, "question_type": "paper_abstract_semantic",
            "category": "paper", "target_entity_type": "paper", "difficulty": "hard",
            "target_entity_iri": pid,
            "answer": c["title"], "answer_type": "literal",
            "text_answer": f"The paper is '{_abbrev(c['title'], 100)}'.",
            "is_answerable": True, "source_fields": ["abstract"],
        }):
            added_hard += 1

    # --- Hard: paper_multi_metadata (author + task + abstract clue) ---
    multi_tmpls = [
        "Which paper by {author} on {task} begins its abstract with: '{frag}'?",
        "What paper authored by {author} covering {task} starts with: '{frag}'?",
        "Which work by {author} about {task} has this abstract opening: '{frag}'?",
    ]
    for c in with_authors_abstract:
        if added_hard >= hard_target:
            break
        pid = c["paper_id"]
        if iri_uses[pid] >= MAX_USES_PER_IRI:
            continue
        if not c["tasks"]:
            continue
        a = c["authors"][0]; t = c["tasks"][0]
        frag = _abstract_fragment(c["abstract"], c["title"], max_chars=80)
        if len(frag) < 20:
            continue
        tmpl = _tmpl(multi_tmpls, pid + "h2")
        q_text = tmpl.format(author=a, task=t, frag=frag)
        if _try_add({
            "question": q_text, "question_type": "paper_multi_metadata",
            "category": "paper", "target_entity_type": "paper", "difficulty": "hard",
            "target_entity_iri": pid,
            "answer": c["title"], "answer_type": "literal",
            "text_answer": f"The paper is '{_abbrev(c['title'], 100)}'.",
            "is_answerable": True, "source_fields": ["authors", "tasks", "abstract"],
        }):
            added_hard += 1

    return out


def generate_dataset_questions(
    dataset_idx: dict, iri_uses: Counter, target: int, used_texts: set
) -> list[dict]:
    out: list[dict] = []

    # Build clue indexes for ambiguity check
    task_to_ids: dict[str, set] = defaultdict(set)
    kw_to_ids:   dict[str, set] = defaultdict(set)
    task_kw_to_ids: dict[tuple, set] = defaultdict(set)

    for did, rec in dataset_idx.items():
        for t in (rec.get("tasks") or []):
            task_to_ids[t].add(did)
        for k in (rec.get("keywords") or []):
            kw_to_ids[k].add(did)
        tasks = [t for t in (rec.get("tasks") or []) if t]
        kws   = [k for k in (rec.get("keywords") or []) if k]
        if tasks and kws:
            task_kw_to_ids[(tasks[0], kws[0])].add(did)

    def _try_add(q_dict: dict) -> bool:
        iri = q_dict["target_entity_iri"]
        if iri_uses[iri] >= MAX_USES_PER_IRI:
            return False
        text = q_dict["question"].strip()
        if text in used_texts:
            return False
        out.append(q_dict)
        iri_uses[iri] += 1
        used_texts.add(text)
        return True

    easy_target  = math.ceil(target * 0.36)
    medium_target = math.ceil(target * 0.40)
    hard_target  = target - easy_target - medium_target

    # Build a stable ordering
    items = sorted(dataset_idx.items(), key=lambda x: x[0])

    # --- Easy: dataset_title_to_entity ---
    title_tmpls = [
        "Which dataset is named '{title}'?",
        "What dataset has the title '{title}'?",
        "Which MLSea dataset corresponds to the name '{title}'?",
        "Find the dataset called '{title}'.",
    ]
    added_easy = 0
    for did, rec in items:
        if added_easy >= easy_target:
            break
        if iri_uses[did] >= MAX_USES_PER_IRI:
            continue
        title = str(rec.get("title") or rec.get("label") or "").strip()
        if not title:
            continue
        tmpl = _tmpl(title_tmpls, did + "e")
        q_text = tmpl.format(title=_abbrev(title, 100))
        if _try_add({
            "question": q_text, "question_type": "dataset_title_to_entity",
            "category": "dataset", "target_entity_type": "dataset", "difficulty": "easy",
            "target_entity_iri": did,
            "answer": title, "answer_type": "literal",
            "text_answer": f"The dataset is '{_abbrev(title, 100)}'.",
            "is_answerable": True, "source_fields": ["title"],
        }):
            added_easy += 1

    # --- Medium: dataset_task_to_dataset ---
    task_tmpls = [
        "Which dataset associated with {task} also has the keyword {kw}?",
        "What dataset is used for {task} and tagged with {kw}?",
        "Which MLSea dataset supports {task} and is associated with {kw}?",
    ]
    task_only_tmpls = [
        "Which dataset is used for {task}?",
        "What dataset supports the {task} benchmark?",
        "Which MLSea dataset is specifically for {task}?",
    ]
    added_med = 0
    for did, rec in items:
        if added_med >= medium_target:
            break
        if iri_uses[did] >= MAX_USES_PER_IRI:
            continue
        tasks = [t for t in (rec.get("tasks") or []) if t]
        kws   = [k for k in (rec.get("keywords") or []) if k]
        title = str(rec.get("title") or rec.get("label") or "").strip()
        if not tasks:
            continue
        t = tasks[0]
        if kws:
            k = kws[0]
            # Ambiguity check: task+keyword combo unique?
            if len(task_kw_to_ids.get((t, k), set())) > 1:
                # Try other pairs
                found = False
                for t2 in tasks[:3]:
                    for k2 in kws[:3]:
                        if len(task_kw_to_ids.get((t2, k2), set())) == 1:
                            t, k = t2, k2; found = True; break
                    if found:
                        break
                if not found:
                    continue
            tmpl = _tmpl(task_tmpls, did + "m")
            q_text = tmpl.format(task=t, kw=k)
            sf = ["tasks", "keywords"]
            qt = "dataset_task_to_dataset"
        else:
            # Use task alone only if unique
            if len(task_to_ids.get(t, set())) > 1:
                continue
            tmpl = _tmpl(task_only_tmpls, did + "m")
            q_text = tmpl.format(task=t)
            sf = ["tasks"]
            qt = "dataset_task_to_dataset"
        if _try_add({
            "question": q_text, "question_type": qt,
            "category": "dataset", "target_entity_type": "dataset", "difficulty": "medium",
            "target_entity_iri": did,
            "answer": title, "answer_type": "literal",
            "text_answer": f"The dataset is '{_abbrev(title, 100)}'.",
            "is_answerable": True, "source_fields": sf,
        }):
            added_med += 1

    # --- Medium: dataset_keyword_to_dataset ---
    kw_tmpls = [
        "Which dataset is associated with the keyword {kw}?",
        "What dataset is tagged with the keyword {kw}?",
        "Which MLSea dataset is specifically linked to {kw}?",
    ]
    for did, rec in items:
        if added_med >= medium_target:
            break
        if iri_uses[did] >= MAX_USES_PER_IRI:
            continue
        kws = [k for k in (rec.get("keywords") or []) if k]
        title = str(rec.get("title") or rec.get("label") or "").strip()
        if not kws:
            continue
        k = kws[0]
        if len(kw_to_ids.get(k, set())) > 1:
            # Try finding unique keyword
            found_k = None
            for k2 in kws:
                if len(kw_to_ids.get(k2, set())) == 1:
                    found_k = k2; break
            if not found_k:
                continue
            k = found_k
        tmpl = _tmpl(kw_tmpls, did + "m2")
        q_text = tmpl.format(kw=k)
        if _try_add({
            "question": q_text, "question_type": "dataset_keyword_to_dataset",
            "category": "dataset", "target_entity_type": "dataset", "difficulty": "medium",
            "target_entity_iri": did,
            "answer": title, "answer_type": "literal",
            "text_answer": f"The dataset is '{_abbrev(title, 100)}'.",
            "is_answerable": True, "source_fields": ["keywords"],
        }):
            added_med += 1

    # --- Hard: dataset_multi_metadata ---
    multi_tmpls = [
        "Which dataset released in {year} is linked to {task} and {kw}?",
        "What dataset from {year} supports {task} and is tagged with {kw}?",
        "Which MLSea dataset issued in {year} covers {task} and {kw}?",
    ]
    for did, rec in items:
        if len(out) >= target:
            break
        if iri_uses[did] >= MAX_USES_PER_IRI:
            continue
        tasks = [t for t in (rec.get("tasks") or []) if t]
        kws   = [k for k in (rec.get("keywords") or []) if k]
        year_raw = str(rec.get("issued_year") or "")[:4]
        title = str(rec.get("title") or rec.get("label") or "").strip()
        if not tasks or not kws or not year_raw.isdigit():
            continue
        t = tasks[0]; k = kws[0]; y = year_raw
        tmpl = _tmpl(multi_tmpls, did + "h")
        q_text = tmpl.format(year=y, task=t, kw=k)
        _try_add({
            "question": q_text, "question_type": "dataset_multi_metadata",
            "category": "dataset", "target_entity_type": "dataset", "difficulty": "hard",
            "target_entity_iri": did,
            "answer": title, "answer_type": "literal",
            "text_answer": f"The dataset is '{_abbrev(title, 100)}'.",
            "is_answerable": True, "source_fields": ["tasks", "keywords", "issued_year"],
        })

    return out


def generate_model_questions(
    model_idx: dict, sibling_idx: dict, iri_uses: Counter, target: int, used_texts: set
) -> list[dict]:
    out: list[dict] = []

    # Repo → model_ids for ambiguity check
    repo_to_ids: dict[str, set] = defaultdict(set)
    for mid, rec in model_idx.items():
        repos = list(dict.fromkeys(
            le["object_label"]
            for le in (rec.get("linked_entities") or [])
            if le.get("predicate_label") == "codeRepository" and le.get("object_label")
        ))
        for r in repos:
            repo_to_ids[r].add(mid)

    def _try_add(q_dict: dict) -> bool:
        iri = q_dict["target_entity_iri"]
        if iri_uses[iri] >= MAX_USES_PER_IRI:
            return False
        text = q_dict["question"].strip()
        if text in used_texts:
            return False
        out.append(q_dict)
        iri_uses[iri] += 1
        used_texts.add(text)
        return True

    easy_target   = math.ceil(target * 0.48)  # ~60/125
    medium_target = math.ceil(target * 0.32)  # ~40/125
    # rest is hard

    items = sorted(model_idx.items(), key=lambda x: x[0])

    # --- Easy: model_name_to_entity ---
    name_tmpls = [
        "Which model is named '{name}'?",
        "What ML model has the label '{name}'?",
        "Which model in MLSea is called '{name}'?",
        "Find the model with the name '{name}'.",
    ]
    added_easy = 0
    for mid, rec in items:
        if added_easy >= easy_target:
            break
        if iri_uses[mid] >= MAX_USES_PER_IRI:
            continue
        label = str(rec.get("label") or rec.get("title") or "").strip()
        if not label:
            continue
        tmpl = _tmpl(name_tmpls, mid + "e")
        q_text = tmpl.format(name=_abbrev(label, 80))
        if _try_add({
            "question": q_text, "question_type": "model_name_to_entity",
            "category": "model", "target_entity_type": "model", "difficulty": "easy",
            "target_entity_iri": mid,
            "answer": label, "answer_type": "literal",
            "text_answer": f"The model is '{label}'.",
            "is_answerable": True, "source_fields": ["label"],
        }):
            added_easy += 1

    # --- Medium: model_repository_to_model ---
    repo_tmpls = [
        "Which model is associated with the {repo} repository?",
        "What model is linked to the {repo} code repository?",
        "Which model has implementation evidence from {repo}?",
        "What ML model is connected to the {repo} GitHub repository?",
    ]
    added_med = 0
    for mid, rec in items:
        if added_med >= medium_target // 2:
            break
        if iri_uses[mid] >= MAX_USES_PER_IRI:
            continue
        repos = list(dict.fromkeys(
            le["object_label"]
            for le in (rec.get("linked_entities") or [])
            if le.get("predicate_label") == "codeRepository" and le.get("object_label")
        ))
        if not repos:
            continue
        label = str(rec.get("label") or "").strip()
        # Find a repo that uniquely (or near-uniquely) identifies this model
        chosen_repo = None
        for r in repos:
            if len(repo_to_ids.get(r, set())) == 1:
                chosen_repo = r; break
        if not chosen_repo:
            # Use first repo but add parent paper clue in text
            chosen_repo = repos[0]
            # Still use it; it's common for models from same paper to share repo
        tmpl = _tmpl(repo_tmpls, mid + "m1")
        q_text = tmpl.format(repo=chosen_repo)
        if _try_add({
            "question": q_text, "question_type": "model_repository_to_model",
            "category": "model", "target_entity_type": "model", "difficulty": "medium",
            "target_entity_iri": mid,
            "answer": label, "answer_type": "literal",
            "text_answer": f"The model is '{label}'.",
            "is_answerable": True, "source_fields": ["linked_entities", "codeRepository"],
        }):
            added_med += 1

    # --- Medium: model_paper_to_model (only when exactly one sibling) ---
    paper_tmpls = [
        "Which model belongs to the '{paper}' work?",
        "What model is the sole variant from the '{paper}' paper?",
        "Which model is associated with the research paper '{paper}'?",
    ]
    for mid, rec in items:
        if added_med >= medium_target:
            break
        if iri_uses[mid] >= MAX_USES_PER_IRI:
            continue
        rest = mid[len(MODEL_PREFIX):]
        parent = rest.rsplit("/", 1)[0] if "/" in rest else ""
        if not parent:
            continue
        siblings = sibling_idx.get(parent, [])
        if len(siblings) != 1:
            continue  # Only use when exactly one model for this paper
        label = str(rec.get("label") or "").strip()
        paper_abbrev = _abbrev(parent, 100)
        tmpl = _tmpl(paper_tmpls, mid + "m2")
        q_text = tmpl.format(paper=paper_abbrev)
        if _try_add({
            "question": q_text, "question_type": "model_paper_to_model",
            "category": "model", "target_entity_type": "model", "difficulty": "medium",
            "target_entity_iri": mid,
            "answer": label, "answer_type": "literal",
            "text_answer": f"The model is '{label}'.",
            "is_answerable": True, "source_fields": ["model_id", "paper_title"],
        }):
            added_med += 1

    # --- Hard: model_multi_metadata (paper title + repo) ---
    multi_tmpls = [
        "Which model from the '{paper}' work is linked to the {repo} repository?",
        "What model belonging to '{paper}' has implementation evidence in {repo}?",
        "Which model from the paper '{paper}' is connected to the {repo} repository?",
    ]
    added_hard = 0
    hard_target_num = target - easy_target - medium_target
    for mid, rec in items:
        if added_hard >= hard_target_num // 2:
            break
        if iri_uses[mid] >= MAX_USES_PER_IRI:
            continue
        rest = mid[len(MODEL_PREFIX):]
        parent = rest.rsplit("/", 1)[0] if "/" in rest else ""
        if not parent:
            continue
        repos = list(dict.fromkeys(
            le["object_label"]
            for le in (rec.get("linked_entities") or [])
            if le.get("predicate_label") == "codeRepository" and le.get("object_label")
        ))
        if not repos:
            continue
        label = str(rec.get("label") or "").strip()
        siblings = sibling_idx.get(parent, [])
        if len(siblings) < 2:
            continue  # family-variant questions need siblings
        # Check that this repo narrows it to this model
        repo = repos[0]
        models_with_this_repo_in_family = [
            s for s in siblings
            if repo in s["repos"]
        ]
        if len(models_with_this_repo_in_family) > 1:
            # Still use it but it might not uniquely identify
            pass
        paper_abbrev = _abbrev(parent, 90)
        tmpl = _tmpl(multi_tmpls, mid + "h1")
        q_text = tmpl.format(paper=paper_abbrev, repo=repo)
        if _try_add({
            "question": q_text, "question_type": "model_multi_metadata",
            "category": "model", "target_entity_type": "model", "difficulty": "hard",
            "target_entity_iri": mid,
            "answer": label, "answer_type": "literal",
            "text_answer": f"The model is '{label}'.",
            "is_answerable": True, "source_fields": ["model_id", "paper_title", "linked_entities"],
        }):
            added_hard += 1

    # --- Hard: model_family_variant (multiple siblings, distinguish by name) ---
    family_tmpls = [
        "Which model variant in the '{paper}' family has the name containing '{hint}'?",
        "What variant from the '{paper}' paper is specifically named with '{hint}'?",
        "Which model in the '{paper}' work is identified by '{hint}' in its name?",
    ]
    for parent, siblings in sorted(sibling_idx.items(), key=lambda x: x[0]):
        if added_hard >= hard_target_num:
            break
        if len(siblings) < 2:
            continue
        for s in siblings:
            mid = s["model_id"]
            if iri_uses[mid] >= MAX_USES_PER_IRI:
                continue
            label = s["label"]
            if not label or len(label) < 3:
                continue
            # Find a distinguishing hint: a unique word in the label
            hint = None
            label_words = label.split()
            for word in label_words:
                if word and not any(
                    word in sibling["label"]
                    for sibling in siblings
                    if sibling["model_id"] != mid
                ):
                    hint = word; break
            if not hint:
                # Use the full label as hint but mark medium
                hint = label
            paper_abbrev = _abbrev(parent, 80)
            tmpl = _tmpl(family_tmpls, mid + "h2")
            q_text = tmpl.format(paper=paper_abbrev, hint=hint)
            if _try_add({
                "question": q_text, "question_type": "model_family_variant",
                "category": "model", "target_entity_type": "model", "difficulty": "hard",
                "target_entity_iri": mid,
                "answer": label, "answer_type": "literal",
                "text_answer": f"The model is '{label}'.",
                "is_answerable": True, "source_fields": ["label", "model_id"],
            }):
                added_hard += 1
                break  # One per family per round

    return out


def _unanswerable_questions() -> list[dict]:
    """Fixed set of 20 plausible but unanswerable questions (10 to keep from existing, 10 new)."""
    new_unans = [
        # Paper style (1 new)
        {
            "question": "Which paper in the MLSea knowledge graph was published with a Creative Commons CC-BY 4.0 license?",
            "question_type": "unanswerable_paper_metadata",
            "category": "paper", "difficulty": "medium",
        },
        # Dataset style (2 new)
        {
            "question": "Which MLSea dataset has exactly 1,234,567 training examples?",
            "question_type": "unanswerable_dataset_metadata",
            "category": "dataset", "difficulty": "hard",
        },
        {
            "question": "Which dataset in the MLSea graph was exclusively collected from hospital patient records in Germany?",
            "question_type": "unanswerable_dataset_metadata",
            "category": "dataset", "difficulty": "medium",
        },
        # Model style (7 new)
        {
            "question": "Which model in the MLSea knowledge graph achieves 100% accuracy on the ImageNet benchmark?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "hard",
        },
        {
            "question": "What is the exact number of parameters of the largest model listed in the MLSea graph?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "hard",
        },
        {
            "question": "Which model variant in MLSea was specifically designed for quantum computing inference?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "medium",
        },
        {
            "question": "Which MLSea model was used in a 2023 Nobel Prize research paper?",
            "question_type": "unanswerable_cross_entity_relation",
            "category": "model", "difficulty": "hard",
        },
        {
            "question": "Which model in MLSea has a carbon footprint of less than 1 kg CO₂ for full training?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "medium",
        },
        {
            "question": "Which model variant listed in MLSea was tested on a proprietary benchmark not publicly available?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "hard",
        },
        {
            "question": "Which model in the MLSea graph requires less than 1 GB of GPU memory for inference?",
            "question_type": "unanswerable_model_metadata",
            "category": "model", "difficulty": "medium",
        },
    ]
    result = []
    for d in new_unans:
        result.append({
            "question": d["question"],
            "question_type": d["question_type"],
            "category": d["category"],
            "target_entity_type": d["category"],
            "difficulty": d["difficulty"],
            "target_entity_iri": None,
            "answer": None,
            "answer_type": "unanswerable",
            "text_answer": "This question is not answerable from the available MLSea records.",
            "is_answerable": False,
            "source_fields": [],
        })
    return result


# ── Good unanswerable category/type fixes ────────────────────────────────────

_GOOD_UNANS_CATEGORIES = {
    "mlsea_q_216": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_217": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_218": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_219": ("dataset", "unanswerable_dataset_metadata"),
    "mlsea_q_220": ("dataset", "unanswerable_dataset_metadata"),
    "mlsea_q_221": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_222": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_223": ("paper", "unanswerable_paper_metadata"),
    "mlsea_q_224": ("dataset", "unanswerable_dataset_metadata"),
    "mlsea_q_225": ("dataset", "unanswerable_dataset_metadata"),
}

_OFFTOPIC_IDS = {"mlsea_q_021", "mlsea_q_022", "mlsea_q_023", "mlsea_q_024", "mlsea_q_025"}


def _fix_unanswerable(q: dict) -> dict | None:
    """Fix schema of a valid unanswerable question. Return None if off-topic."""
    qid = q.get("id", "")
    if qid in _OFFTOPIC_IDS:
        return None
    q = dict(q)
    cat_fix = _GOOD_UNANS_CATEGORIES.get(qid)
    if cat_fix:
        q["category"], q["question_type"] = cat_fix
    else:
        # Try to infer category from question text
        text = q.get("question", "").lower()
        if "paper" in text or "publication" in text or "doi" in text:
            q["category"] = "paper"
        elif "dataset" in text:
            q["category"] = "dataset"
        elif "model" in text:
            q["category"] = "model"
        else:
            q["category"] = "paper"
    q["target_entity_type"] = q["category"]
    q["answer_type"] = "unanswerable"
    q["target_entity_iri"] = None
    q["answer"] = None
    q["is_answerable"] = False
    q["source_fields"] = []
    if not q.get("difficulty"):
        q["difficulty"] = "hard"
    q["text_answer"] = "This question is not answerable from the available MLSea records."
    return q


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=== MLSea Question Dataset Generator ===", flush=True)

    # ── 0. Backup ────────────────────────────────────────────────────────────
    if not BACKUP_PATH.exists():
        shutil.copy2(QUESTIONS_PATH, BACKUP_PATH)
        print(f"[Backup] Created: {BACKUP_PATH.name}", flush=True)
    else:
        print(f"[Backup] Already exists: {BACKUP_PATH.name}", flush=True)

    # ── A. Load existing questions ───────────────────────────────────────────
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        old_questions: list[dict] = json.load(fh)
    print(f"[Load] {len(old_questions)} existing questions", flush=True)

    # Extract all used IRIs
    all_old_iris = {
        normalize_iri(str(q.get("target_entity_iri") or ""))
        for q in old_questions
        if q.get("is_answerable", True) and q.get("target_entity_iri")
    }

    # ── B. Load canonical indexes ────────────────────────────────────────────
    paper_id_set, paper_candidates = load_paper_index(used_iris=set())
    dataset_idx = load_dataset_index()
    model_idx, sibling_idx = load_model_index()

    # Also build a paper candidate lookup by ID for REPLACE
    paper_cand_by_id = {c["paper_id"]: c for c in paper_candidates}

    # ── C. Audit existing questions ──────────────────────────────────────────
    print("[Audit] Processing existing questions ...", flush=True)
    iri_uses: Counter = Counter()   # normalized_iri → count
    kept_paper    : list[dict] = []
    kept_dataset  : list[dict] = []
    kept_model    : list[dict] = []
    kept_unanswer : list[dict] = []
    audit_log: list[dict] = []
    missing_targets: list[dict] = []
    seen_texts: set[str] = set()   # deduplicate question texts during audit

    for q in old_questions:
        qid = q.get("id", "?")
        is_ans = q.get("is_answerable", True)

        # ── Unanswerable handling ─────────────────────────────────────────
        if not is_ans:
            fixed = _fix_unanswerable(q)
            if fixed is None:
                audit_log.append({"id": qid, "status": "DISCARD", "reason": "off-topic unanswerable"})
                continue
            kept_unanswer.append(fixed)
            audit_log.append({"id": qid, "status": "REPAIR", "reason": "unanswerable schema fix"})
            continue

        # ── Answerable: infer category ────────────────────────────────────
        raw_cat = q.get("category", "")
        cat = CATEGORY_REMAP.get(raw_cat, raw_cat)
        if cat not in VALID_CATEGORIES:
            cat = category_from_iri(str(q.get("target_entity_iri") or ""))
        if not cat:
            audit_log.append({"id": qid, "status": "DISCARD", "reason": "cannot determine category"})
            missing_targets.append({
                "id": qid, "category": raw_cat,
                "target_entity_iri": q.get("target_entity_iri"),
                "reason": "cannot determine category", "outcome": "removed",
            })
            continue

        iri_raw = normalize_iri(str(q.get("target_entity_iri") or ""))
        qt_orig = q.get("question_type", "")
        qt = QTYPE_REMAP.get(qt_orig, qt_orig)

        # ── Validate IRI in canonical ─────────────────────────────────────
        iri_valid = (
            (cat == "paper"   and iri_raw in paper_id_set) or
            (cat == "dataset" and iri_raw in dataset_idx)  or
            (cat == "model"   and iri_raw in model_idx)
        )
        if not iri_valid:
            # Try URL-decoded form
            iri_dec = unquote(iri_raw)
            iri_valid2 = (
                (cat == "paper"   and iri_dec in paper_id_set) or
                (cat == "dataset" and iri_dec in dataset_idx)  or
                (cat == "model"   and iri_dec in model_idx)
            )
            if iri_valid2:
                iri_raw = iri_dec
            else:
                audit_log.append({"id": qid, "status": "DISCARD", "reason": f"IRI not in canonical ({iri_raw[:60]})"})
                missing_targets.append({
                    "id": qid, "category": cat,
                    "target_entity_iri": q.get("target_entity_iri"),
                    "reason": "IRI not found in canonical records", "outcome": "removed",
                })
                continue

        # ── Property-extraction → REPLACE ─────────────────────────────────
        if qt_orig in PROPERTY_TYPES or qt_orig == "paper_to_implementation":
            # Try to REPLACE with entity-retrieval question for same IRI
            if iri_uses[iri_raw] >= MAX_USES_PER_IRI:
                audit_log.append({"id": qid, "status": "DISCARD", "reason": f"property question; IRI already used"})
                continue
            rep_q = None
            if cat == "paper":
                rep_q = _build_replacement_paper(q, iri_raw, paper_id_set, paper_candidates, iri_uses)
            elif cat == "dataset":
                rep_q = _build_replacement_dataset(q, iri_raw, dataset_idx, iri_uses)
            if rep_q:
                rep_text = rep_q.get("question", "").strip()
                if rep_text in seen_texts:
                    audit_log.append({"id": qid, "status": "DISCARD", "reason": "replacement has duplicate text"})
                    continue
                seen_texts.add(rep_text)
                rep_q["target_entity_type"] = cat
                iri_uses[iri_raw] += 1
                if cat == "paper":
                    kept_paper.append(rep_q)
                elif cat == "dataset":
                    kept_dataset.append(rep_q)
                audit_log.append({"id": qid, "status": "REPLACE",
                                  "reason": f"property question replaced with {rep_q['question_type']}"})
            else:
                audit_log.append({"id": qid, "status": "DISCARD", "reason": "property question; no replacement possible"})
            continue

        # ── Valid entity-retrieval question → KEEP/REPAIR ─────────────────
        if iri_uses[iri_raw] >= MAX_USES_PER_IRI:
            audit_log.append({"id": qid, "status": "DISCARD", "reason": "IRI already used twice"})
            continue

        q2 = dict(q)
        q2["question_type"]      = qt
        q2["category"]           = cat
        q2["target_entity_type"] = cat
        q2["target_entity_iri"]  = iri_raw
        q2["is_answerable"]      = True
        if q2.get("difficulty") not in VALID_DIFFICULTIES:
            q2["difficulty"] = _infer_difficulty(qt)
        if q2.get("answer_type") not in {"literal", "entity"}:
            q2["answer_type"] = "literal"
        if not q2.get("source_fields"):
            q2["source_fields"] = _infer_source_fields(qt)
        if not q2.get("text_answer"):
            ans = q2.get("answer", "")
            q2["text_answer"] = f"The answer is {ans}." if ans else "See canonical record."

        # Deduplicate by question text
        q_text = q2.get("question", "").strip()
        if q_text in seen_texts:
            audit_log.append({"id": qid, "status": "DISCARD", "reason": "duplicate question text"})
            continue
        seen_texts.add(q_text)

        iri_uses[iri_raw] += 1
        if cat == "paper":
            kept_paper.append(q2)
        elif cat == "dataset":
            kept_dataset.append(q2)
        else:
            kept_model.append(q2)

        status = "REPAIR" if (
            q.get("question_type") != qt or
            q.get("category") != cat or
            not q.get("source_fields") or
            q.get("difficulty") not in VALID_DIFFICULTIES
        ) else "KEEP"
        audit_log.append({"id": qid, "status": status, "reason": "valid entity-retrieval"})

    print(f"[Audit] KEEP+REPAIR: paper={len(kept_paper)}, dataset={len(kept_dataset)}, "
          f"model={len(kept_model)}, unanswerable={len(kept_unanswer)}", flush=True)

    # ── D. Generate additional questions ──────────────────────────────────────
    # Track IRIs used by salvaged questions (paper_candidates are fresh)
    # For generation, only use IRIs NOT already in iri_uses
    def _fresh_candidates(cands: list) -> list:
        return [c for c in cands if iri_uses.get(c["paper_id"], 0) < MAX_USES_PER_IRI]

    fresh_paper = _fresh_candidates(paper_candidates)

    need_paper   = max(0, TARGET_PAPER   - len(kept_paper))
    need_dataset = max(0, TARGET_DATASET - len(kept_dataset))
    need_model   = max(0, TARGET_MODEL   - len(kept_model))

    # Collect existing question texts to prevent duplicates across kept + generated
    # seen_texts already has all audited question texts; start from that
    used_texts: set[str] = set(seen_texts)

    print(f"[Generate] Need paper={need_paper}, dataset={need_dataset}, model={need_model}", flush=True)

    new_paper   = generate_paper_questions(fresh_paper, iri_uses, need_paper, {}, used_texts)
    print(f"[Generate] Generated {len(new_paper)} paper questions", flush=True)
    new_dataset = generate_dataset_questions(dataset_idx, iri_uses, need_dataset, used_texts)
    print(f"[Generate] Generated {len(new_dataset)} dataset questions", flush=True)
    new_model   = generate_model_questions(model_idx, sibling_idx, iri_uses, need_model, used_texts)
    print(f"[Generate] Generated {len(new_model)} model questions", flush=True)

    # ── E. Unanswerable: fill to 20 ──────────────────────────────────────────
    new_unans = _unanswerable_questions()
    all_unanswerable = kept_unanswer + new_unans
    # Trim if over 20
    if len(all_unanswerable) > TARGET_UNANSWERABLE:
        all_unanswerable = all_unanswerable[:TARGET_UNANSWERABLE]

    # ── F. Combine and trim to targets ────────────────────────────────────────
    all_paper   = (kept_paper   + new_paper)  [:TARGET_PAPER]
    all_dataset = (kept_dataset + new_dataset)[:TARGET_DATASET]
    all_model   = (kept_model   + new_model)  [:TARGET_MODEL]
    all_questions = all_paper + all_dataset + all_model + all_unanswerable

    print(f"[Final] paper={len(all_paper)}, dataset={len(all_dataset)}, "
          f"model={len(all_model)}, unanswerable={len(all_unanswerable)}, "
          f"total={len(all_questions)}", flush=True)

    # ── G. Assign sequential IDs ──────────────────────────────────────────────
    for i, q in enumerate(all_questions, start=1):
        q["id"] = f"mlsea_q_{i:03d}"

    # ── H. Write dataset ──────────────────────────────────────────────────────
    with QUESTIONS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(all_questions, fh, ensure_ascii=False, indent=2)
    print(f"[Write] {QUESTIONS_PATH.name} written ({len(all_questions)} questions)", flush=True)

    # ── I. Reports ────────────────────────────────────────────────────────────
    _write_summary(all_questions, audit_log, new_paper, new_dataset, new_model, all_unanswerable)
    _write_audit(old_questions, audit_log, all_questions)
    if missing_targets:
        with MISSING_PATH.open("w", encoding="utf-8") as fh:
            json.dump(missing_targets, fh, ensure_ascii=False, indent=2)
        print(f"[Write] {MISSING_PATH.name} ({len(missing_targets)} entries)", flush=True)

    print("=== Done ===", flush=True)
    return 0


def _write_summary(questions: list, audit_log: list, new_paper: list,
                   new_dataset: list, new_model: list, unans: list) -> None:
    answerable = [q for q in questions if q.get("is_answerable", True)]
    unanswerable = [q for q in questions if not q.get("is_answerable", True)]
    cat_ctr   = Counter(q.get("category") for q in answerable)
    diff_ctr  = Counter(q.get("difficulty") for q in answerable)
    qt_ctr    = Counter(q.get("question_type") for q in questions)
    iri_list  = [q.get("target_entity_iri") for q in answerable if q.get("target_entity_iri")]
    iri_ctr   = Counter(iri_list)
    unique_iris = len(iri_ctr)
    max_iri   = max(iri_ctr.values()) if iri_ctr else 0
    avg_iri   = len(iri_list) / unique_iris if unique_iris else 0
    iri_by_cat: dict[str, set] = defaultdict(set)
    for q in answerable:
        c = q.get("category"); iri = q.get("target_entity_iri")
        if c and iri:
            iri_by_cat[c].add(iri)

    status_ctr = Counter(e["status"] for e in audit_log)
    n_kept    = status_ctr["KEEP"]
    n_repair  = status_ctr["REPAIR"]
    n_replace = status_ctr["REPLACE"]
    n_discard = status_ctr["DISCARD"]
    n_new     = len(new_paper) + len(new_dataset) + len(new_model) + sum(
        1 for e in audit_log if e["status"] == "REPLACE"
    )

    lines = [
        "# ML Questions Dataset Summary",
        "",
        f"Generated by `scripts/generate_question_dataset.py`.",
        "",
        "## Counts",
        f"- Total questions: {len(questions)}",
        f"- Answerable: {len(answerable)}",
        f"- Unanswerable: {len(unanswerable)}",
        "",
        "## Category Distribution (answerable)",
        *(f"- {c}: {n}" for c, n in sorted(cat_ctr.items())),
        "",
        "## Difficulty Distribution (answerable)",
        *(f"- {d}: {n}" for d, n in sorted(diff_ctr.items())),
        "",
        "## Question Type Distribution",
        *(f"- {qt}: {n}" for qt, n in sorted(qt_ctr.items())),
        "",
        "## Gold Target IRI Diversity",
        f"- Unique target IRIs (answerable): {unique_iris}",
        *(f"- Unique {c} IRIs: {len(s)}" for c, s in sorted(iri_by_cat.items())),
        f"- Average questions per IRI: {avg_iri:.2f}",
        f"- Maximum questions per IRI: {max_iri}",
        "",
        "## Audit of Original Dataset",
        f"- Questions kept (KEEP): {n_kept}",
        f"- Questions repaired (REPAIR): {n_repair}",
        f"- Questions replaced (REPLACE): {n_replace}",
        f"- Questions discarded (DISCARD): {n_discard}",
        f"- New questions generated: {len(new_paper) + len(new_dataset) + len(new_model)}",
        "",
        "## Limitations",
        "- Model canonical records contain no description, tasks, datasets, or metrics.",
        "  Model questions use only: model name, parent paper title (from model_id), and repository names.",
        "  Types `model_description_to_model`, `model_task_to_model`, `model_dataset_to_model` are not generated.",
        "- Dataset canonical records contain no description field.",
        "  Dataset questions use: title, tasks, keywords, issued_year, linked_entities.",
        "  Type `dataset_description_to_title` is not generated.",
        "- Paper abstract-based questions use actual abstract fragments (not paraphrased).",
        "  These test semantic retrieval capability against partial abstract text.",
        "- Hard model questions rely on family-variant disambiguation and paper+repo combinations.",
        "  If a family's repo links do not distinguish siblings, those questions are skipped.",
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[Write] {SUMMARY_PATH.name}", flush=True)


def _write_audit(old_questions: list, audit_log: list, new_questions: list) -> None:
    status_ctr = Counter(e["status"] for e in audit_log)
    discard_reasons = Counter(e["reason"] for e in audit_log if e["status"] == "DISCARD")
    replace_examples = [e for e in audit_log if e["status"] == "REPLACE"][:5]
    discard_examples = [e for e in audit_log if e["status"] == "DISCARD"][:5]

    lines = [
        "# ML Questions Dataset Audit Report",
        "",
        f"Original dataset: {len(old_questions)} questions",
        "",
        "## Audit Results",
        f"- KEEP (valid, no changes): {status_ctr['KEEP']}",
        f"- REPAIR (valid, schema fixed): {status_ctr['REPAIR']}",
        f"- REPLACE (property→entity-retrieval): {status_ctr['REPLACE']}",
        f"- DISCARD (removed): {status_ctr['DISCARD']}",
        "",
        "## Discard Reasons",
        *(f"- {r}: {n}" for r, n in discard_reasons.most_common()),
        "",
        "## Example Replaced Questions",
        *(f"- {e['id']}: {e['reason']}" for e in replace_examples),
        "",
        "## Example Discarded Questions",
        *(f"- {e['id']}: {e['reason']}" for e in discard_examples),
        "",
        "## Final Distribution",
        f"- Total: {len(new_questions)}",
        *(f"- {c}: {n}" for c, n in Counter(q.get("category") for q in new_questions).most_common()),
    ]
    with AUDIT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[Write] {AUDIT_PATH.name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
