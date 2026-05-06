"""Build FAISS vector stores for all LectureRAG-75 domain transcripts.

Reads merged transcripts from data/lecture_rag_75/transcripts/<domain>.txt
and saves FAISS stores to data/lecture_rag_75/stores/<domain>/.

Usage:
    python scripts/build_domain_stores.py             # all domains
    python scripts/build_domain_stores.py --domain generative_ai
"""

import argparse
import json
import os

TRANSCRIPTS_DIR = "data/lecture_rag_75/transcripts"
STORES_DIR = "data/lecture_rag_75/stores"
EXISTING_TRANSCRIPTS = {
    "generative_ai": ["data/transcript_7.txt"],
}
CONFIG_PATH = "data/lecture_rag_75/corpus_config.json"


def build_store(domain: str, transcript_path: str, stores_dir: str,
                chunk_size: int = 600, overlap: int = 80):
    from src.build_vectorstore import build_faiss_store
    out_dir = os.path.join(stores_dir, domain)
    meta_path = os.path.join(out_dir, "meta.pkl")
    if os.path.exists(meta_path):
        print(f"  [skip] Store already exists: {out_dir}")
        return out_dir

    if not os.path.exists(transcript_path):
        print(f"  [skip] Transcript not found: {transcript_path}")
        return None

    print(f"  Building FAISS store for {domain} (chunk_size={chunk_size}) ...")
    build_faiss_store(transcript_path, out_dir, chunk_size=chunk_size, overlap=overlap)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Build FAISS stores for LectureRAG-75")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--transcripts-dir", default=TRANSCRIPTS_DIR)
    parser.add_argument("--stores-dir", default=STORES_DIR)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    os.makedirs(args.stores_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    # Existing domains (local transcripts)
    for dom, dcfg in config.get("existing_domains", {}).items():
        if args.domain and dom != args.domain:
            continue
        transcripts = dcfg.get("transcript_files", [])
        # Prefer the first existing transcript
        for tf in transcripts:
            if os.path.exists(tf):
                print(f"\nDomain: {dom}")
                build_store(dom, tf, args.stores_dir)
                break
        else:
            print(f"[skip] No transcript found for {dom}")

    # New domains (merged transcripts)
    for dom in config.get("domains", {}):
        if args.domain and dom != args.domain:
            continue
        # Support both flat (transcripts/domain.txt) and nested (transcripts/domain/domain.txt)
        transcript_path = os.path.join(args.transcripts_dir, f"{dom}.txt")
        if not os.path.exists(transcript_path):
            transcript_path = os.path.join(args.transcripts_dir, dom, f"{dom}.txt")
        print(f"\nDomain: {dom}")
        build_store(dom, transcript_path, args.stores_dir)

    print("\nDone. Run scripts/build_lecture_rag_75.py to assemble the final QA benchmark.")


if __name__ == "__main__":
    main()
