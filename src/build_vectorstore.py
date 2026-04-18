from sentence_transformers import SentenceTransformer
import pickle
import os

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def create_embeddings(transcript_path, vector_store_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks from transcript.")
    embeddings = model.encode(chunks, show_progress_bar=True)
    vector_store = {"chunks": chunks, "embeddings": embeddings}
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    with open(vector_store_path, "wb") as f:
        pickle.dump(vector_store, f)
    print(f"Vector store saved to {vector_store_path}")
