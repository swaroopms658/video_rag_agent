"""Merge per-domain FAISS stores into a single combined store.

After running build_domain_stores.py for each domain, run this script to merge
all domain FAISS indexes into data/lecture_rag_75/combined/ so that
cold_start_eval.py can retrieve across all domains with a single index.

Usage:
    python scripts/build_combined_store.py
    python scripts/build_combined_store.py --stores-dir data/lecture_rag_75/stores \
                                            --out data/lecture_rag_75/combined
"""

import argparse
import os
import pickle
from pathlib import Path


STORES_DIR = "data/lecture_rag_75/stores"
OUT_DIR = "data/lecture_rag_75/combined"


def merge_stores(stores_dir: str, out_dir: str):
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise ImportError("pip install faiss-cpu numpy")

    from src.agent import make_chunk_id

    store_dirs = [p for p in sorted(Path(stores_dir).iterdir())
                  if p.is_dir() and (p / "index.faiss").exists()]

    if not store_dirs:
        print(f"No domain stores found in {stores_dir}")
        return

    print(f"Merging {len(store_dirs)} domain store(s):")
    all_chunks, all_ids, all_domains = [], [], []
    dim = None
    all_embeddings = []

    for sdir in store_dirs:
        domain = sdir.name
        idx = faiss.read_index(str(sdir / "index.faiss"))
        with open(sdir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        chunks = meta["chunks"]
        n = len(chunks)
        print(f"  {domain}: {n} chunks")

        # Reconstruct embeddings from the index
        if dim is None:
            dim = idx.d
        vecs = np.zeros((n, idx.d), dtype="float32")
        for i in range(n):
            idx.reconstruct(i, vecs[i])

        all_embeddings.append(vecs)
        all_chunks.extend(chunks)
        all_ids.extend([make_chunk_id(c) for c in chunks])
        all_domains.extend([domain] * n)

    combined_embs = np.vstack(all_embeddings)
    print(f"\nCombined: {len(all_chunks)} chunks, dim={dim}")

    combined_idx = faiss.IndexFlatIP(dim)
    combined_idx.add(combined_embs)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(combined_idx, os.path.join(out_dir, "index.faiss"))

    meta = {
        "chunks": all_chunks,
        "ids": all_ids,
        "domains": all_domains,
        "timestamps": [None] * len(all_chunks),
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Combined store saved -> {out_dir}  ({len(all_chunks)} chunks)")
    print(f"Domain breakdown: { {d: all_domains.count(d) for d in sorted(set(all_domains))} }")


def main():
    parser = argparse.ArgumentParser(description="Merge domain FAISS stores")
    parser.add_argument("--stores-dir", default=STORES_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    merge_stores(args.stores_dir, args.out)


if __name__ == "__main__":
    main()
