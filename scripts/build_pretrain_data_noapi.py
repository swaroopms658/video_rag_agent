"""Build (query, positive, negative) pretraining triples WITHOUT any API calls.

Uses simple sentence-extraction heuristics + BM25 hard negatives across
all transcripts in data/itma_pretrain/transcripts/. Each triple is tagged
with its source transcript so the curriculum trainer can build helpful
(same-source) and unhelpful (cross-source) memory.

Heuristics for query generation:
  1. From each chunk, extract the first declarative sentence.
  2. Optionally rewrite as a "What is X" question by stripping noun phrases.
  3. Filter out very short or very generic sentences.

Wall-clock: <30 seconds for ~5 transcripts. No internet access required
beyond the initial transcript download.

Usage:
    python scripts/build_pretrain_data_noapi.py
    python scripts/build_pretrain_data_noapi.py --max-chunks-per-file 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MIN_QUERY_LEN = 25
MAX_QUERY_LEN = 200
MIN_CHUNK_LEN = 200
HARD_NEG_PER_QUERY = 1
EASY_NEG_FRAC = 0.3   # fraction of triples whose negative comes from a *different* transcript


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, pos = [], 0
    while pos < len(text):
        end = min(pos + size, len(text))
        c = text[pos:end].strip()
        if len(c) >= MIN_CHUNK_LEN:
            chunks.append(c)
        pos += size - overlap
    return chunks


def load_transcript(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\d+\.\d+\s+", "", line)
        line = re.sub(r"^\[\d+:\d{2}:\d{2}.*?]\s*", "", line)
        if line:
            lines.append(line)
    return " ".join(lines)


GENERIC_STARTERS = {
    "this", "that", "these", "those", "it", "they", "we", "you", "i",
    "in", "on", "the", "a", "an", "for", "of", "to", "and", "but",
    "however", "moreover", "furthermore", "also",
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def is_good_query(s: str) -> bool:
    if not (MIN_QUERY_LEN <= len(s) <= MAX_QUERY_LEN):
        return False
    first = s.split()[0].lower() if s.split() else ""
    if first in GENERIC_STARTERS:
        return False
    if s.endswith(":"):
        return False
    return True


def make_query(chunk: str) -> str | None:
    """Pick the first sentence in the chunk that looks like a self-contained
    fact, then rephrase it as a 'What is X?' question."""
    for sent in split_sentences(chunk):
        if not is_good_query(sent):
            continue
        # If the sentence already looks like a question, use it directly.
        if sent.endswith("?"):
            return sent
        # "X is Y" → "What is X?"
        m = re.match(r"([A-Z][\w\- ]{2,40})\s+is\s+", sent)
        if m:
            term = m.group(1).strip()
            return f"What is {term}?"
        # Fallback: turn the sentence into a question by prepending
        return f"What does the following statement describe? {sent}"
    return None


def bm25_hard_negative(chunk: str, candidates: list[str], rng: random.Random) -> str | None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        others = [c for c in candidates if c != chunk]
        return rng.choice(others) if others else None

    def tok(t):
        return re.findall(r"[a-z0-9]+", t.lower())

    pool = [c for c in candidates if c != chunk]
    if not pool:
        return None
    bm = BM25Okapi([tok(c) for c in pool])
    scores = bm.get_scores(tok(chunk))
    top = int(max(range(len(scores)), key=lambda i: scores[i]))
    return pool[top]


def build_triples(
    transcripts_dir: str,
    out_path: str,
    max_chunks_per_file: int = 100,
    seed: int = 42,
):
    rng = random.Random(seed)
    transcripts_dir_p = Path(transcripts_dir)
    files = sorted(transcripts_dir_p.rglob("*.txt"))
    if not files:
        print(f"No .txt files in {transcripts_dir}")
        return 0

    print(f"Found {len(files)} transcript file(s)")

    # Load chunks per source
    src_chunks: dict[str, list[str]] = {}
    for f in files:
        text = load_transcript(f)
        chunks = chunk_text(text)
        if not chunks:
            print(f"  [skip] {f.name}: no chunks")
            continue
        if len(chunks) > max_chunks_per_file:
            step = len(chunks) / max_chunks_per_file
            chunks = [chunks[int(i * step)] for i in range(max_chunks_per_file)]
        src_chunks[f.stem] = chunks
        print(f"  {f.stem}: {len(chunks)} chunks")

    # Cross-source pool for easy negatives
    all_chunks_with_src: list[tuple[str, str]] = []
    for src, chunks in src_chunks.items():
        all_chunks_with_src.extend((src, c) for c in chunks)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    skipped = 0
    with out_path_p.open("w", encoding="utf-8") as f:
        for src, chunks in src_chunks.items():
            for chunk in chunks:
                q = make_query(chunk)
                if not q:
                    skipped += 1
                    continue

                # Decide easy (cross-source) or hard (same-source) negative
                if rng.random() < EASY_NEG_FRAC:
                    other = [(s, c) for s, c in all_chunks_with_src if s != src]
                    if other:
                        _, neg = rng.choice(other)
                    else:
                        neg = bm25_hard_negative(chunk, chunks, rng)
                else:
                    neg = bm25_hard_negative(chunk, chunks, rng)

                if neg is None:
                    skipped += 1
                    continue

                f.write(json.dumps({
                    "query": q,
                    "positive": chunk,
                    "negative": neg,
                    "source": src,
                }, ensure_ascii=False) + "\n")
                n_total += 1

    print(f"\nWrote {n_total} triples to {out_path}  (skipped {skipped})")
    print(f"Sources: {list(src_chunks.keys())}")
    return n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default="data/itma_pretrain/transcripts")
    parser.add_argument("--out", default="data/itma_pretrain/triples.jsonl")
    parser.add_argument("--max-chunks-per-file", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_triples(
        args.transcripts, args.out, args.max_chunks_per_file, args.seed,
    )


if __name__ == "__main__":
    main()
