"""Auto-draft QA candidates from lecture transcripts using Groq Llama-3.1-8B.

Outputs a JSONL file of UNVERIFIED candidates for manual review and editing.
Each line is one candidate:
  {
    "question": "...",
    "candidate_answer": "...",
    "source_chunk": "...",       <- the transcript chunk it came from
    "domain": "machine_learning",
    "difficulty": "factual",    <- factual | inferential | multi-hop
    "verified": false            <- set to true after manual review
  }

Usage:
    # Draft from a single transcript file
    python scripts/draft_qa.py --transcript data/lecture_rag_75/transcripts/machine_learning.txt \\
                                --domain machine_learning \\
                                --output data/lecture_rag_75/candidates/machine_learning.jsonl

    # Draft all domains listed in corpus_config.json
    python scripts/draft_qa.py --all

    # Expand existing GenAI domain
    python scripts/draft_qa.py --transcript data/lecture_transcript.txt \\
                                --domain generative_ai \\
                                --output data/lecture_rag_75/candidates/generative_ai.jsonl
"""

import argparse
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "data/lecture_rag_75/corpus_config.json"
CANDIDATES_DIR = "data/lecture_rag_75/candidates"

CHUNK_SIZE = 800   # chars per context chunk fed to LLM
CHUNK_OVERLAP = 100
CANDIDATES_PER_CHUNK = 2   # how many QA pairs to request per chunk


PROMPT_TEMPLATE = """\
You are building a question-answering benchmark from a lecture transcript.

TRANSCRIPT CHUNK:
\"\"\"{chunk}\"\"\"

Generate exactly {n} question-answer pairs from this chunk. Follow these rules:
1. Questions must be answerable SOLELY from the chunk above — no outside knowledge.
2. Mix difficulty: include at least one FACTUAL question (directly stated) and one INFERENTIAL question (requires combining two pieces of information from the chunk).
3. Answers should be 1-3 sentences, precise, and grounded in the chunk.
4. Do NOT copy the question word-for-word into the answer.

Respond in this EXACT JSON format (a JSON array, nothing else):
[
  {{
    "question": "...",
    "answer": "...",
    "difficulty": "factual"
  }},
  {{
    "question": "...",
    "answer": "...",
    "difficulty": "inferential"
  }}
]"""


def _parse_retry_after(text: str) -> float:
    """Extract wait seconds from Groq 429 message, e.g. 'try again in 240ms' or '1.5s'."""
    m = re.search(r"try again in ([\d.]+)(m?s)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if m.group(2).lower() == "ms" else val
    return 5.0


def groq_generate(prompt, api_key, model="llama-3.1-8b-instant", max_tokens=600,
                  max_retries=6):
    import json as _json
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    payload = _json.dumps(data, ensure_ascii=False).encode("utf-8")
    for attempt in range(max_retries):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            data=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            wait = _parse_retry_after(resp.text) + 0.5 * (attempt + 1)
            print(f"  [rate limit] sleeping {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    raise RuntimeError(f"Groq rate limit: exceeded {max_retries} retries")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple overlapping character-level chunker. Returns list of strings."""
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        chunks.append(text[pos:end].strip())
        pos += size - overlap
    return [c for c in chunks if len(c) > 100]  # skip tiny tail chunks


def parse_llm_json(raw):
    """Extract JSON array from LLM output, which may include markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


def load_transcript_text(path):
    """Load a transcript file, stripping timestamp prefixes if present."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip leading timestamp (e.g. "12.34  " or "[0:00:12 --> 0:00:15]")
            line = re.sub(r"^\d+\.\d+\s+", "", line)
            line = re.sub(r"^\[\d+:\d{2}:\d{2}.*?]\s*", "", line)
            if line:
                lines.append(line)
    return " ".join(lines)


def draft_domain(transcript_path, domain, output_path, api_key,
                 n_per_chunk=CANDIDATES_PER_CHUNK, max_chunks=20, delay=2.0):
    """Generate QA candidates from a transcript and write to output_path JSONL."""
    print(f"\nDrafting QA candidates for domain: {domain}")
    print(f"  Transcript: {transcript_path}")

    text = load_transcript_text(transcript_path)
    chunks = chunk_text(text)

    # Sample evenly across the transcript if there are more chunks than max_chunks
    if len(chunks) > max_chunks:
        step = len(chunks) / max_chunks
        chunks = [chunks[int(i * step)] for i in range(max_chunks)]

    print(f"  {len(chunks)} chunks selected (max_chunks={max_chunks})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(chunk=chunk, n=n_per_chunk)
            try:
                raw = groq_generate(prompt, api_key)
                pairs = parse_llm_json(raw)
            except Exception as e:
                print(f"  [chunk {i}] Error: {e}")
                time.sleep(delay * 2)
                continue

            for pair in pairs:
                q = str(pair.get("question", "")).strip()
                a = str(pair.get("answer", "")).strip()
                d = str(pair.get("difficulty", "factual")).strip()
                if not q or not a:
                    continue
                record = {
                    "question": q,
                    "candidate_answer": a,
                    "source_chunk": chunk[:300],  # first 300 chars for traceability
                    "domain": domain,
                    "difficulty": d if d in ("factual", "inferential", "multi-hop") else "factual",
                    "verified": False,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

            print(f"  [chunk {i}/{len(chunks)}] +{len(pairs)} candidates  (total={total})")
            time.sleep(delay)

    print(f"\n  Done: {total} candidates written to {output_path}")
    print(f"  NEXT STEP: Open {output_path}, review each entry, edit answers, set 'verified': true.")
    return total


def main():
    parser = argparse.ArgumentParser(description="Draft QA candidates from lecture transcripts")
    parser.add_argument("--transcript", default=None, help="Path to a single transcript file")
    parser.add_argument("--domain", default=None, help="Domain label for this transcript")
    parser.add_argument("--output", default=None, help="Output JSONL path for candidates")
    parser.add_argument("--all", action="store_true", help="Draft all domains in corpus_config.json")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--candidates-dir", default=CANDIDATES_DIR)
    parser.add_argument("--max-chunks", type=int, default=20,
                        help="Max chunks to sample per transcript (default: 20 → ~40 candidates)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between Groq API calls")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set in environment / .env file")
        return

    if args.transcript and args.domain:
        out = args.output or os.path.join(args.candidates_dir, f"{args.domain}.jsonl")
        draft_domain(args.transcript, args.domain, out, api_key,
                     max_chunks=args.max_chunks, delay=args.delay)
        return

    if args.all:
        with open(args.config) as f:
            config = json.load(f)

        # Existing GenAI domain from local transcripts
        for domain_name, dcfg in config.get("existing_domains", {}).items():
            for tf in dcfg["transcript_files"]:
                if os.path.exists(tf):
                    out = os.path.join(args.candidates_dir, f"{domain_name}.jsonl")
                    draft_domain(tf, domain_name, out, api_key,
                                 max_chunks=args.max_chunks, delay=args.delay)
                    break
            else:
                print(f"[skip] No transcript found for existing domain: {domain_name}")

        # New domains from fetched YouTube transcripts
        for domain_name in config.get("domains", {}):
            merged = os.path.join("data/lecture_rag_75/transcripts", f"{domain_name}.txt")
            if not os.path.exists(merged):
                print(f"[skip] Transcript not found for {domain_name}: {merged}")
                print(f"       Run: python scripts/fetch_transcripts.py --domain {domain_name}")
                continue
            out = os.path.join(args.candidates_dir, f"{domain_name}.jsonl")
            draft_domain(merged, domain_name, out, api_key,
                         max_chunks=args.max_chunks, delay=args.delay)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
