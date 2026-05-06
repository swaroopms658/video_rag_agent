"""Fetch YouTube lecture transcripts for LectureRAG-75 corpus.

Usage:
    python scripts/fetch_transcripts.py                        # fetch all domains
    python scripts/fetch_transcripts.py --domain machine_learning
    python scripts/fetch_transcripts.py --domain machine_learning --video-id abc123

Transcripts are saved to data/lecture_rag_75/transcripts/<domain>/<video_id>.txt
Each line: "<timestamp_sec>  <text>"

Requirements:
    pip install youtube-transcript-api
"""

import argparse
import json
import os
import time


CONFIG_PATH = "data/lecture_rag_75/corpus_config.json"
OUT_DIR = "data/lecture_rag_75/transcripts"


def fetch_video_transcript(video_id, lang="en", retries=3, base_delay=3.0, cookies=None):
    """Download transcript for a single YouTube video. Returns list of (start_sec, text) tuples.

    cookies: path to a Netscape-format cookies.txt file (e.g. exported via 'Get cookies.txt' browser ext)
             This bypasses YouTube's IP rate-limit (HTTP 429).
    """
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

    kwargs = {}
    if cookies:
        try:
            import http.cookiejar
            jar = http.cookiejar.MozillaCookieJar(cookies)
            jar.load(ignore_discard=True, ignore_expires=True)
            import requests
            session = requests.Session()
            session.cookies = jar
            kwargs["http_client"] = session
        except Exception as e:
            print(f"  [warn] Could not load cookies from {cookies}: {e}")

    api = YouTubeTranscriptApi(**kwargs)
    for attempt in range(1, retries + 1):
        try:
            transcript = api.fetch(video_id, languages=[lang, "en-US", "en-GB"])
            return [(seg.start, seg.text) for seg in transcript]
        except TranscriptsDisabled:
            print(f"  [error] Transcripts are disabled for {video_id}")
            return []
        except Exception as e:
            err_str = str(e)
            if "disabled" in err_str.lower() or "age-restricted" in err_str.lower():
                print(f"  [error] Transcripts unavailable for {video_id}: {err_str[:80]}")
                return []
            if attempt < retries:
                wait = base_delay * attempt
                print(f"  [retry {attempt}/{retries}] {video_id} — waiting {wait:.0f}s ...")
                time.sleep(wait)
            else:
                # Final attempt: try via list() → individual transcript fetch
                print(f"  [warn] Direct fetch failed for {video_id}, trying list() fallback ...")
                try:
                    tl = api.list(video_id)
                    t = next(
                        (x for x in tl if x.language_code.startswith("en")),
                        None
                    )
                    if t is None:
                        print(f"  [error] No English transcript listed for {video_id}")
                        return []
                    time.sleep(base_delay)
                    fetched = t.fetch()
                    return [(seg.start, seg.text) for seg in fetched]
                except Exception as e2:
                    print(f"  [error] All fetch attempts failed for {video_id}: {e2}")
                    return []
    return []


def save_transcript(segments, out_path):
    """Write transcript to file: one line per segment, '<start_sec>  <text>'."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for start, text in segments:
            f.write(f"{start:.2f}  {text}\n")
    print(f"  Saved {len(segments)} segments -> {out_path}")


def fetch_domain(domain_name, video_ids, out_dir, delay=1.5, cookies=None):
    domain_dir = os.path.join(out_dir, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    saved = []
    for vid in video_ids:
        out_path = os.path.join(domain_dir, f"{vid}.txt")
        if os.path.exists(out_path):
            print(f"  [skip] {vid} already exists at {out_path}")
            saved.append(out_path)
            continue
        print(f"  Fetching {vid} ...")
        segments = fetch_video_transcript(vid, cookies=cookies)
        if segments:
            save_transcript(segments, out_path)
            saved.append(out_path)
        time.sleep(delay)  # polite rate limiting
    return saved


def concatenate_domain_transcripts(domain_name, out_dir):
    """Merge all per-video transcripts for a domain into a single file."""
    domain_dir = os.path.join(out_dir, domain_name)
    merged_path = os.path.join(out_dir, f"{domain_name}.txt")
    files = sorted(f for f in os.listdir(domain_dir) if f.endswith(".txt"))
    if not files:
        return None
    with open(merged_path, "w", encoding="utf-8") as out:
        for fname in files:
            with open(os.path.join(domain_dir, fname), encoding="utf-8") as inp:
                out.write(f"# === Video: {fname[:-4]} ===\n")
                out.write(inp.read())
                out.write("\n")
    print(f"  Merged {len(files)} files -> {merged_path}")
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Fetch lecture transcripts from YouTube")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--domain", default=None, help="Fetch a specific domain only")
    parser.add_argument("--video-id", default=None, help="Fetch a single video (requires --domain)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument("--cookies", default=None,
                        help="Path to Netscape cookies.txt (exported from browser). "
                             "Use this to bypass YouTube HTTP 429 rate-limits.")
    args = parser.parse_args()

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("ERROR: youtube-transcript-api not installed.")
        print("Run: pip install youtube-transcript-api")
        return

    with open(args.config) as f:
        config = json.load(f)

    domains_to_fetch = config["domains"]
    if args.domain:
        if args.domain not in domains_to_fetch:
            print(f"Domain '{args.domain}' not found. Available: {list(domains_to_fetch)}")
            return
        domains_to_fetch = {args.domain: domains_to_fetch[args.domain]}

    for domain_name, domain_cfg in domains_to_fetch.items():
        video_ids = domain_cfg["video_ids"]
        if args.video_id:
            video_ids = [args.video_id]

        print(f"\nFetching domain: {domain_name} ({len(video_ids)} videos)")
        print(f"  {domain_cfg['description']}")
        saved = fetch_domain(domain_name, video_ids, args.out,
                             delay=args.delay, cookies=args.cookies)
        if saved:
            merged = concatenate_domain_transcripts(domain_name, args.out)
            if merged:
                print(f"  -> Combined transcript: {merged}")

    print("\nDone. Next step: run scripts/draft_qa.py to generate QA candidates.")


if __name__ == "__main__":
    main()
