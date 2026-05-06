import re
import pickle
import os
from sentence_transformers import SentenceTransformer

_TS_RE = re.compile(
    r'^\[(\d+):(\d{2}):(\d{2}\.\d+)\s*-->\s*(\d+):(\d{2}):(\d{2}\.\d+)\]\s*(.*)'
)


def _parse_time(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


def _load_segments(path):
    """Return list of (text, start_sec, end_sec). Falls back to (text, None, None) for plain text."""
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _TS_RE.match(line)
            if m:
                start = _parse_time(m.group(1), m.group(2), m.group(3))
                end = _parse_time(m.group(4), m.group(5), m.group(6))
                segments.append((m.group(7).strip(), start, end))
            else:
                segments.append((line, None, None))
    return segments


def _chunk_segments(segments, chunk_size=1000, overlap=100):
    """Produce overlapping text chunks with (start_sec, end_sec) timestamp spans."""
    has_ts = any(s[1] is not None for s in segments)

    # Build a flat character buffer tracking each segment's char range and timestamps
    full_text = ""
    seg_map = []  # (char_start, char_end, start_sec, end_sec)
    for text, start, end in segments:
        cs = len(full_text)
        full_text += text + " "
        seg_map.append((cs, len(full_text), start, end))

    chunks, timestamps = [], []
    pos = 0
    while pos < len(full_text):
        end_pos = min(pos + chunk_size, len(full_text))
        chunk = full_text[pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
            if has_ts:
                in_range = [(s, e) for cs, ce, s, e in seg_map
                            if s is not None and e is not None and cs < end_pos and ce > pos]
                if in_range:
                    timestamps.append((min(s for s, _ in in_range), max(e for _, e in in_range)))
                else:
                    timestamps.append(None)
            else:
                timestamps.append(None)
        pos += chunk_size - overlap

    return chunks, timestamps


def create_embeddings(transcript_path, vector_store_path):
    """Legacy: saves pickle with embeddings array. Still supported for backward compat."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    segments = _load_segments(transcript_path)
    chunks, chunk_timestamps = _chunk_segments(segments)
    print(f"Created {len(chunks)} chunks from transcript.")
    embeddings = model.encode(chunks, show_progress_bar=True)
    vector_store = {
        "chunks": chunks,
        "embeddings": embeddings,
        "timestamps": chunk_timestamps,
    }
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    with open(vector_store_path, "wb") as f:
        pickle.dump(vector_store, f)
    print(f"Vector store saved to {vector_store_path}")


def build_faiss_store(transcript_path, output_dir, model_name='all-MiniLM-L6-v2',
                      chunk_size=1000, overlap=100):
    """Build a FAISS IndexFlatIP store (cosine via L2-normalised inner product).

    Saves two files into output_dir:
      - index.faiss   : FAISS index (float32, L2-normalised embeddings)
      - meta.pkl      : {'chunks': [...], 'timestamps': [...]} — no raw embeddings
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise ImportError("pip install faiss-cpu  to use build_faiss_store()")

    model = SentenceTransformer(model_name)
    segments = _load_segments(transcript_path)
    chunks, timestamps = _chunk_segments(segments, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks from transcript.")

    embeddings = model.encode(chunks, show_progress_bar=True).astype("float32")
    # L2-normalise so IndexFlatIP == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    meta = {"chunks": chunks, "timestamps": timestamps}
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"FAISS store saved to {output_dir}  ({len(chunks)} chunks, dim={dim})")
