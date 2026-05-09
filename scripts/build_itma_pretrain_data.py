"""Build synthetic triplet data for one-time ITMA pretraining.

Generates (query, positive_context, hard_negative) triples from held-out
lecture transcripts (DISJOINT from the LectureRAG-75 eval set).

The held-out corpus is data/itma_pretrain/transcripts/ (fetched separately,
never included in eval).

Steps:
  1. Chunk each held-out transcript.
  2. Use Groq Llama-3.1-8B to synthesise 1–2 queries per chunk (positive).
  3. Use BM25 to find hard negatives (top-ranked non-positive chunks).
  4. Optionally synthesise mixed-quality memory contexts for each triple.
  5. Write triples to data/itma_pretrain/triples.jsonl

Usage:
    python scripts/build_itma_pretrain_data.py
    python scripts/build_itma_pretrain_data.py --transcripts data/itma_pretrain/transcripts \
                                                --out data/itma_pretrain/triples.jsonl \
                                                --max-chunks 200 --synth-memory

Requirements:
    GROQ_API_KEY in .env
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PRETRAIN_DIR = "data/itma_pretrain"
TRANSCRIPTS_DIR = os.path.join(PRETRAIN_DIR, "transcripts")
OUT_JSONL = os.path.join(PRETRAIN_DIR, "triples.jsonl")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MAX_CHUNKS_PER_FILE = 50
QUERIES_PER_CHUNK = 2
HARD_NEG_PER_QUERY = 3
SYNTH_MEMORY_ENTRIES = 4   # how many memory entries to synthesise per triple
GROQ_DELAY = 1.5


QUERY_PROMPT = """\
You are generating training data for a retrieval model.

PASSAGE:
\"\"\"{chunk}\"\"\"

Generate {n} short, specific questions that are fully answerable from this passage.
Output as a JSON array of strings (questions only, no answers):
["question 1", "question 2"]"""


MEMORY_PROMPT = """\
Given the following question and its correct passage, generate {n} related but DISTINCT questions
that might appear in the same session. For each, indicate whether the passage above would be helpful (1)
or not helpful (0) for that related question.

ORIGINAL QUESTION: {question}
PASSAGE: {passage}

Output as a JSON array:
[{{"question": "...", "helpful": 1}}, ...]"""


def groq_generate(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant",
                  max_tokens: int = 400) -> str:
    import json as _json
    import requests
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=_json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    if resp.status_code == 429:
        raise RuntimeError(f"RATE_LIMIT:{resp.text[:100]}")
    raise RuntimeError(f"Groq error {resp.status_code}: {resp.text[:200]}")


def hf_generate(prompt: str, max_tokens: int = 400) -> str:
    """HF InferenceClient fallback for query generation."""
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise RuntimeError("HF_API_KEY not set")
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="auto", api_key=hf_key)
    result = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    return result.choices[0].message.content


def generate(prompt: str, api_key: str, max_tokens: int = 400) -> str:
    """Try Groq first; fall back to HF on rate limit."""
    try:
        return groq_generate(prompt, api_key, max_tokens=max_tokens)
    except RuntimeError as e:
        if "RATE_LIMIT" in str(e):
            time.sleep(2)
            return hf_generate(prompt, max_tokens=max_tokens)
        raise


def parse_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, pos = [], 0
    while pos < len(text):
        end = min(pos + size, len(text))
        c = text[pos:end].strip()
        if len(c) > 80:
            chunks.append(c)
        pos += size - overlap
    return chunks


def load_transcript(path: str) -> str:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^\d+\.\d+\s+", "", line)
            line = re.sub(r"^\[\d+:\d{2}:\d{2}.*?]\s*", "", line)
            if line:
                lines.append(line)
    return " ".join(lines)


def find_hard_negatives(positive_chunk: str, all_chunks: list[str], k: int = 3) -> list[str]:
    """BM25-based hard negatives: top-k chunks that are most similar but not the positive."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # Fallback: random negatives
        others = [c for c in all_chunks if c != positive_chunk]
        return random.sample(others, min(k, len(others)))

    def tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    candidates = [c for c in all_chunks if c != positive_chunk]
    if not candidates:
        return []
    bm25 = BM25Okapi([tok(c) for c in candidates])
    query_tokens = tok(positive_chunk)
    scores = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [candidates[i] for i in top_idx]


def build_triples(
    transcripts_dir: str,
    out_path: str,
    api_key: str,
    max_chunks: int = MAX_CHUNKS_PER_FILE,
    synth_memory: bool = False,
    delay: float = GROQ_DELAY,
    seed: int = 42,
):
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    transcript_files = list(Path(transcripts_dir).rglob("*.txt"))
    if not transcript_files:
        print(f"No transcript files found in {transcripts_dir}")
        print("Fetch held-out transcripts first (separate from eval domains).")
        return 0

    print(f"Found {len(transcript_files)} transcript file(s) in {transcripts_dir}")
    total = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for tf in transcript_files:
            print(f"\nProcessing: {tf.name}")
            text = load_transcript(str(tf))
            all_chunks = chunk_text(text)
            if not all_chunks:
                print(f"  [skip] No chunks extracted")
                continue

            # Sample up to max_chunks evenly
            if len(all_chunks) > max_chunks:
                step = len(all_chunks) / max_chunks
                sampled = [all_chunks[int(i * step)] for i in range(max_chunks)]
            else:
                sampled = all_chunks

            print(f"  {len(sampled)} chunks to process")

            for i, chunk in enumerate(sampled, 1):
                # Generate query candidates
                prompt = QUERY_PROMPT.format(chunk=chunk[:500], n=QUERIES_PER_CHUNK)
                try:
                    raw = generate(prompt, api_key)
                    queries = parse_json(raw)
                    if not isinstance(queries, list):
                        queries = []
                    queries = [str(q).strip() for q in queries if q][:QUERIES_PER_CHUNK]
                except Exception as e:
                    print(f"  [chunk {i}] Query gen error: {e}")
                    time.sleep(delay * 2)
                    continue

                hard_negs = find_hard_negatives(chunk, all_chunks, k=HARD_NEG_PER_QUERY)

                for query in queries:
                    if not query:
                        continue
                    for neg in hard_negs[:1]:  # one neg per query for speed
                        triple: dict = {
                            "query": query,
                            "positive": chunk,
                            "negative": neg,
                            "source": tf.stem,
                        }

                        # Optionally synthesise memory context
                        if synth_memory and rng.random() < 0.7:
                            mem_prompt = MEMORY_PROMPT.format(
                                question=query, passage=chunk[:400],
                                n=SYNTH_MEMORY_ENTRIES,
                            )
                            try:
                                mem_raw = groq_generate(mem_prompt, api_key, max_tokens=300)
                                mem_items = parse_json(mem_raw)
                                if isinstance(mem_items, list) and mem_items:
                                    triple["memory_queries"] = [
                                        m.get("question", "") for m in mem_items
                                        if isinstance(m, dict)
                                    ]
                                    # For memory contexts, use the positive chunk for all
                                    # (simplification — in practice would vary)
                                    triple["memory_contexts"] = [
                                        chunk[:400] for _ in triple["memory_queries"]
                                    ]
                                    triple["memory_rewards"] = [
                                        float(m.get("helpful", 1)) * 2 - 1  # 0/1 → -1/+1
                                        for m in mem_items
                                        if isinstance(m, dict)
                                    ]
                                    time.sleep(delay)
                            except Exception:
                                pass  # synth memory is optional

                        out_f.write(json.dumps(triple, ensure_ascii=False) + "\n")
                        total += 1

                print(f"  [chunk {i}/{len(sampled)}]  total triples so far: {total}")
                time.sleep(delay)

    print(f"\nDone. {total} triples written -> {out_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Build ITMA pretraining triples")
    parser.add_argument("--transcripts", default=TRANSCRIPTS_DIR)
    parser.add_argument("--out", default=OUT_JSONL)
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS_PER_FILE)
    parser.add_argument("--synth-memory", action="store_true",
                        help="Generate synthetic memory contexts (2× more Groq calls)")
    parser.add_argument("--delay", type=float, default=GROQ_DELAY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        return

    build_triples(
        transcripts_dir=args.transcripts,
        out_path=args.out,
        api_key=api_key,
        max_chunks=args.max_chunks,
        synth_memory=args.synth_memory,
        delay=args.delay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
