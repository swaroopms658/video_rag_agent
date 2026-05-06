"""Assemble the final LectureRAG-75 QA benchmark from verified candidate files.

Steps:
  1. Load all verified candidates from data/lecture_rag_75/candidates/*.jsonl
  2. For each QA pair, find the best-matching chunk in the domain's FAISS store
     using BERTScore-F1 between the gold answer and each chunk — this chunk
     becomes the gold context span.
  3. Assign gold_context_ids (sha1_16 of the matching chunk).
  4. Write the final eval file: data/lecture_rag_75/qa.jsonl
  5. Write a stratified 60/20/20 split: data/lecture_rag_75/splits.json

Usage:
    # Build from all domains (requires FAISS stores to exist)
    python scripts/build_lecture_rag_75.py

    # Build only from GenAI domain
    python scripts/build_lecture_rag_75.py --domain generative_ai

    # Dry run — report counts, skip file writes
    python scripts/build_lecture_rag_75.py --dry-run

Requirements:
    pip install bert-score sentence-transformers faiss-cpu
"""

import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path


CANDIDATES_DIR = "data/lecture_rag_75/candidates"
STORES_ROOT = "data/lecture_rag_75/stores"   # built by build_vectorstore.py
OUT_QA = "data/lecture_rag_75/qa.jsonl"
OUT_SPLITS = "data/lecture_rag_75/splits.json"
SEED = 42
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.60, 0.20, 0.20
BERTSCORE_THRESHOLD = 0.55   # minimum BERTScore-F1 to accept a chunk as gold


def load_verified_candidates(candidates_dir: str, domain: str | None = None) -> list[dict]:
    items = []
    path = Path(candidates_dir)
    for jsonl in sorted(path.glob("*.jsonl")):
        dom = jsonl.stem.rstrip("_2").rstrip("_1")  # handle generative_ai_2 → generative_ai
        if domain and dom != domain and jsonl.stem != domain:
            continue
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if item.get("verified"):
                    # Use domain from filename if not in item
                    if "domain" not in item:
                        item["domain"] = jsonl.stem
                    items.append(item)
    return items


def load_chunks_for_domain(domain: str, stores_root: str) -> tuple[list[str], list[str]]:
    """Returns (chunks, chunk_ids) from the FAISS meta.pkl for this domain."""
    from src.agent import make_chunk_id
    meta_path = os.path.join(stores_root, domain, "meta.pkl")
    if not os.path.exists(meta_path):
        return [], []
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    chunks = meta["chunks"]
    ids = [make_chunk_id(c) for c in chunks]
    return chunks, ids


def find_gold_context(answer: str, chunks: list[str], chunk_ids: list[str],
                      threshold: float = BERTSCORE_THRESHOLD,
                      source_chunk: str | None = None) -> str | None:
    """Return chunk_id of the gold context for this QA pair.

    Priority:
      1. Exact source_chunk match — if source_chunk is provided (from draft_qa),
         find the chunk that shares the most text overlap with it. This is more
         reliable than BERTScore since we know the original context verbatim.
      2. BERTScore semantic match — used when source_chunk is absent.
      3. Word-overlap fallback if BERTScore is unavailable.
    """
    if not chunks:
        return None

    from src.agent import make_chunk_id as _make_id

    # Priority 1: source_chunk direct match
    if source_chunk and source_chunk.strip():
        src_id = _make_id(source_chunk)
        # Check if any stored chunk has the same ID (exact match after normalisation)
        for cid, chunk in zip(chunk_ids, chunks):
            if cid == src_id:
                return cid
        # Fuzzy fallback: highest word-overlap with source_chunk
        src_words = set(re.findall(r"[a-z0-9]+", source_chunk.lower()))
        best_idx, best_score = 0, 0.0
        for i, chunk in enumerate(chunks):
            chunk_words = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            if not chunk_words:
                continue
            overlap = len(src_words & chunk_words) / len(src_words | chunk_words)
            if overlap > best_score:
                best_score, best_idx = overlap, i
        if best_score >= 0.3:
            return chunk_ids[best_idx]
        # If very low overlap, fall through to BERTScore on the answer

    # Priority 2: BERTScore on the answer text
    from src.eval_utils import bertscore_f1
    refs = chunks
    preds = [answer] * len(chunks)
    try:
        scores = bertscore_f1(preds, refs, lang="en")
    except Exception as e:
        print(f"    [warn] BERTScore failed: {e} — falling back to word-overlap")
        ans_words = set(re.findall(r"[a-z0-9]+", answer.lower()))
        best_idx, best_score = 0, 0.0
        for i, chunk in enumerate(chunks):
            chunk_words = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            if not chunk_words:
                continue
            overlap = len(ans_words & chunk_words) / len(ans_words | chunk_words)
            if overlap > best_score:
                best_score, best_idx = overlap, i
        return chunk_ids[best_idx] if best_score >= 0.1 else None

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    if best_score < threshold:
        return None
    return chunk_ids[best_idx]


def stratified_split(items: list[dict], train_r=TRAIN_RATIO, dev_r=DEV_RATIO,
                     seed=SEED) -> dict[str, list[int]]:
    """Stratify by domain, return indices per split."""
    rng = random.Random(seed)
    domains = sorted(set(it["domain"] for it in items))
    train_ids, dev_ids, test_ids = [], [], []

    for dom in domains:
        idxs = [i for i, it in enumerate(items) if it["domain"] == dom]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, round(n * train_r))
        n_dev = max(1, round(n * dev_r))
        train_ids.extend(idxs[:n_train])
        dev_ids.extend(idxs[n_train:n_train + n_dev])
        test_ids.extend(idxs[n_train + n_dev:])

    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def build(domain: str | None = None, dry_run: bool = False,
          candidates_dir: str = CANDIDATES_DIR,
          stores_root: str = STORES_ROOT,
          out_qa: str = OUT_QA,
          out_splits: str = OUT_SPLITS,
          bertscore_threshold: float = BERTSCORE_THRESHOLD):

    print("Loading verified candidates ...")
    candidates = load_verified_candidates(candidates_dir, domain=domain)
    print(f"  Found {len(candidates)} verified QA pairs")

    if not candidates:
        print("No verified candidates found. Run draft_qa.py and set 'verified': true manually.")
        return

    # Group by domain for store loading
    by_domain: dict[str, list[dict]] = {}
    for item in candidates:
        by_domain.setdefault(item["domain"], []).append(item)

    final_items = []
    for dom, items in sorted(by_domain.items()):
        print(f"\nDomain: {dom}  ({len(items)} candidates)")
        chunks, chunk_ids = load_chunks_for_domain(dom, stores_root)
        if not chunks:
            print(f"  [warn] No FAISS store found at {stores_root}/{dom}/meta.pkl — "
                  f"gold_context_ids will be empty. Build the store first.")
            for item in items:
                final_items.append(_make_qa_item(item, gold_ctx_id=None))
            continue

        print(f"  Loaded {len(chunks)} chunks from FAISS store")
        n_matched = 0
        for item in items:
            answer = item.get("candidate_answer", "")
            source_chunk = item.get("source_chunk", "")
            gold_id = find_gold_context(answer, chunks, chunk_ids,
                                        threshold=bertscore_threshold,
                                        source_chunk=source_chunk)
            if gold_id:
                n_matched += 1
            final_items.append(_make_qa_item(item, gold_ctx_id=gold_id))

        print(f"  {n_matched}/{len(items)} QA pairs matched to gold context chunk")

    if dry_run:
        print(f"\n[dry-run] Would write {len(final_items)} QA pairs to {out_qa}")
        splits = stratified_split(final_items)
        for split_name, idxs in splits.items():
            print(f"  {split_name}: {len(idxs)} items")
        return

    os.makedirs(os.path.dirname(out_qa), exist_ok=True)
    with open(out_qa, "w", encoding="utf-8") as f:
        for item in final_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(final_items)} QA pairs -> {out_qa}")

    splits = stratified_split(final_items)
    split_data = {
        split_name: [final_items[i]["id"] for i in idxs]
        for split_name, idxs in splits.items()
    }
    with open(out_splits, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)
    print(f"Wrote splits -> {out_splits}")
    for split_name, ids in split_data.items():
        print(f"  {split_name}: {len(ids)} items")


def _make_qa_item(candidate: dict, gold_ctx_id: str | None) -> dict:
    import hashlib
    q = candidate.get("question", "").strip()
    uid = hashlib.sha1(q.encode()).hexdigest()[:16]
    return {
        "id": uid,
        "question": q,
        "ground_truth_answer": candidate.get("candidate_answer", "").strip(),
        "domain": candidate.get("domain", "unknown"),
        "difficulty": candidate.get("difficulty", "factual"),
        "gold_context_ids": [gold_ctx_id] if gold_ctx_id else [],
        "source_chunk": candidate.get("source_chunk", "")[:300],
    }


def main():
    parser = argparse.ArgumentParser(description="Build LectureRAG-75 final QA benchmark")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--candidates-dir", default=CANDIDATES_DIR)
    parser.add_argument("--stores-root", default=STORES_ROOT)
    parser.add_argument("--out-qa", default=OUT_QA)
    parser.add_argument("--out-splits", default=OUT_SPLITS)
    parser.add_argument("--bertscore-threshold", type=float, default=BERTSCORE_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build(
        domain=args.domain,
        dry_run=args.dry_run,
        candidates_dir=args.candidates_dir,
        stores_root=args.stores_root,
        out_qa=args.out_qa,
        out_splits=args.out_splits,
        bertscore_threshold=args.bertscore_threshold,
    )


if __name__ == "__main__":
    main()
