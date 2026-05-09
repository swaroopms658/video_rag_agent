"""Expand LectureRAG-75 benchmark from ~115 to 400+ items.

Two-stage pipeline:
  Stage 1 — Generate: run draft_qa.py at higher chunk budget (max_chunks=50, 3/chunk)
  Stage 2 — Verify:   call Groq to verify each candidate, resolve gold_context_ids,
             deduplicate against existing qa.jsonl, append verified items.
  Stage 3 — Splits:   regenerate train/dev/test splits (60/20/20).

Usage:
    python scripts/expand_benchmark.py            # full pipeline (generate + verify)
    python scripts/expand_benchmark.py --verify-only  # skip generation, just verify existing candidates
    python scripts/expand_benchmark.py --gen-only     # just generate new candidates, no verification
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.agent import make_chunk_id

QA_PATH        = "data/lecture_rag_75/qa.jsonl"
SPLITS_PATH    = "data/lecture_rag_75/splits.json"
CANDIDATES_DIR = "data/lecture_rag_75/candidates"
STORES_DIR     = "data/lecture_rag_75/stores"
CONFIG_PATH    = "data/lecture_rag_75/corpus_config.json"
TARGET_ITEMS   = 400


VERIFY_PROMPT = """\
You are a quality-control judge for an educational QA benchmark.

SOURCE CHUNK (from a lecture transcript):
\"\"\"{chunk}\"\"\"

CANDIDATE QUESTION: {question}
CANDIDATE ANSWER: {answer}

Judge this QA pair on TWO criteria:
1. ANSWERABLE: Is the question fully answerable using ONLY the source chunk above (no outside knowledge needed)?
2. CORRECT: Is the candidate answer factually correct and grounded in the source chunk?

Respond in this exact JSON (nothing else):
{{"answerable": true/false, "correct": true/false, "reason": "one sentence"}}"""


def _parse_retry_after(text: str) -> float:
    """Extract wait seconds from Groq 429 message, e.g. 'try again in 240ms' or '1.5s'."""
    m = re.search(r"try again in ([\d.]+)(m?s)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if m.group(2).lower() == "ms" else val
    return 5.0


def groq_generate(prompt, api_key, model="llama-3.1-8b-instant", max_tokens=200,
                  max_retries=8):
    import json as _json
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.1}
    payload = _json.dumps(data, ensure_ascii=False).encode("utf-8")
    for attempt in range(max_retries):
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, data=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            wait = _parse_retry_after(resp.text) + 0.5 * (attempt + 1)
            print(f"  [rate limit] sleeping {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq error {resp.status_code}: {resp.text}")
    raise RuntimeError(f"Groq rate limit: exceeded {max_retries} retries")


def parse_verify_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"answerable": False, "correct": False, "reason": "parse error"}


def load_existing_qa(qa_path: str) -> tuple[list[dict], set[str]]:
    items, questions = [], set()
    if not os.path.exists(qa_path):
        return items, questions
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items.append(item)
                questions.add(item["question"].strip().lower())
    return items, questions


def load_store_chunks(domain: str) -> list[str]:
    """Load all chunks from the FAISS meta.pkl for a domain."""
    meta_path = os.path.join(STORES_DIR, domain, "meta.pkl")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta.get("chunks", [])


def resolve_gold_ids(source_chunk: str, domain: str, max_candidates: int = 3) -> list[str]:
    """Find chunk IDs in the domain store that best match source_chunk."""
    chunks = load_store_chunks(domain)
    if not chunks:
        return []

    # Normalize source_chunk for comparison
    src_norm = " ".join(source_chunk.lower().split())
    prefix = src_norm[:80]

    matches = []
    for chunk in chunks:
        c_norm = " ".join(chunk.lower().split())
        if prefix in c_norm or c_norm[:80] in src_norm:
            matches.append(make_chunk_id(chunk))
        if len(matches) >= max_candidates:
            break

    # Fallback: substring overlap
    if not matches:
        src_words = set(src_norm.split())
        best = sorted(chunks, key=lambda c: len(set(c.lower().split()) & src_words),
                      reverse=True)[:1]
        matches = [make_chunk_id(c) for c in best]

    return matches[:max_candidates]


def new_item_id(question: str) -> str:
    return hashlib.sha1(question.strip().lower().encode()).hexdigest()[:16]


def generate_candidates(api_key: str, max_chunks: int = 50, n_per_chunk: int = 3):
    """Run draft_qa across all domains at higher chunk budget."""
    from scripts.draft_qa import draft_domain
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    for domain_name, dcfg in config.get("existing_domains", {}).items():
        for tf in dcfg.get("transcript_files", []):
            if os.path.exists(tf):
                out = os.path.join(CANDIDATES_DIR, f"{domain_name}_expanded.jsonl")
                draft_domain(tf, domain_name, out, api_key,
                             n_per_chunk=n_per_chunk, max_chunks=max_chunks, delay=3.0)
                break

    for domain_name in config.get("domains", {}):
        merged = f"data/lecture_rag_75/transcripts/{domain_name}/{domain_name}.txt"
        if not os.path.exists(merged):
            # Try flat path
            merged = f"data/lecture_rag_75/transcripts/{domain_name}.txt"
        if not os.path.exists(merged):
            print(f"[skip] transcript not found for {domain_name}")
            continue
        out = os.path.join(CANDIDATES_DIR, f"{domain_name}_expanded.jsonl")
        draft_domain(merged, domain_name, out, api_key,
                     n_per_chunk=n_per_chunk, max_chunks=max_chunks, delay=1.5)


def verify_candidates(api_key: str, qa_path: str = QA_PATH,
                      candidates_dir: str = CANDIDATES_DIR,
                      delay: float = 12.0) -> list[dict]:
    """Auto-verify all unverified candidates. Returns list of new verified items."""
    existing_items, existing_questions = load_existing_qa(qa_path)
    print(f"Existing qa.jsonl: {len(existing_items)} items")

    candidate_files = sorted(
        f for f in os.listdir(candidates_dir) if f.endswith(".jsonl")
    )

    new_items = []
    for cfile in candidate_files:
        cpath = os.path.join(candidates_dir, cfile)
        candidates = []
        with open(cpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        domain_new = 0
        for cand in candidates:
            q = cand.get("question", "").strip()
            a = cand.get("candidate_answer", "").strip()
            src = cand.get("source_chunk", "").strip()
            domain = cand.get("domain", "")
            difficulty = cand.get("difficulty", "factual")

            if not q or not a or not domain:
                continue

            # Skip duplicates
            if q.lower() in existing_questions:
                continue

            # Skip if already human-verified true in the file
            already_verified = cand.get("verified", False)

            if not already_verified:
                # Call Groq to verify
                prompt = VERIFY_PROMPT.format(chunk=src[:600], question=q, answer=a)
                try:
                    raw = groq_generate(prompt, api_key)
                    verdict = parse_verify_response(raw)
                    time.sleep(delay)
                except Exception as e:
                    print(f"  [verify error] {e}")
                    continue

                if not (verdict.get("answerable") and verdict.get("correct")):
                    continue

            # Resolve gold context IDs
            gold_ids = resolve_gold_ids(src, domain)
            if not gold_ids:
                print(f"  [no gold ids] {q[:60]}...")
                continue

            item = {
                "id": new_item_id(q),
                "question": q,
                "ground_truth_answer": a,
                "domain": domain,
                "difficulty": difficulty,
                "gold_context_ids": gold_ids,
                "source_chunk": src[:300],
            }
            new_items.append(item)
            existing_questions.add(q.lower())
            domain_new += 1

        print(f"  {cfile}: +{domain_new} new verified items")

    return new_items


def regenerate_splits(items: list[dict], splits_path: str,
                      train_frac=0.60, dev_frac=0.20, seed=42):
    rng = random.Random(seed)
    ids = [item["id"] for item in items]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    splits = {
        "train": ids[:n_train],
        "dev": ids[n_train:n_train + n_dev],
        "test": ids[n_train + n_dev:],
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits: {len(splits['train'])} train / {len(splits['dev'])} dev / {len(splits['test'])} test")
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--gen-only", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--n-per-chunk", type=int, default=3)
    parser.add_argument("--target", type=int, default=TARGET_ITEMS)
    parser.add_argument("--verify-delay", type=float, default=12.0,
                        help="Seconds between verify API calls (default: 12 = safe under 6000 TPM)")
    parser.add_argument("--qa", default=QA_PATH)
    parser.add_argument("--splits", default=SPLITS_PATH)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    if not args.verify_only:
        print("=== Stage 1: Generating candidates ===")
        generate_candidates(api_key, max_chunks=args.max_chunks, n_per_chunk=args.n_per_chunk)

    if not args.gen_only:
        print("\n=== Stage 2: Verifying candidates ===")
        new_items = verify_candidates(api_key, qa_path=args.qa, delay=args.verify_delay)

        if not new_items:
            print("No new items verified.")
            return

        # Append to qa.jsonl
        existing_items, _ = load_existing_qa(args.qa)
        all_items = existing_items + new_items
        print(f"\nTotal items: {len(existing_items)} existing + {len(new_items)} new = {len(all_items)}")

        with open(args.qa, "w", encoding="utf-8") as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Written {len(all_items)} items -> {args.qa}")

        print("\n=== Stage 3: Regenerating splits ===")
        regenerate_splits(all_items, args.splits)

        if len(all_items) < args.target:
            print(f"\nNote: {len(all_items)} items < target {args.target}. "
                  f"Run again with --max-chunks {args.max_chunks + 20} to generate more.")


if __name__ == "__main__":
    main()
