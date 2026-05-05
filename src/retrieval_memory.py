import json
import os
from sentence_transformers import SentenceTransformer, util
from src.agent import make_chunk_id

FEEDBACK_FILE = "data/rl_feedback.json"

class RetrievalMemory:
    def __init__(self, encoder=None):
        self.memory = []
        self.encoder = encoder if encoder is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.load_memory()

    def load_memory(self):
        """Loads valid feedback (reward=1) from the log file and batch-encodes queries."""
        if not os.path.exists(FEEDBACK_FILE):
            return

        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("reward") == 1:
                        self.memory.append(entry)
                except:
                    continue

        if self.memory:
            queries = [m["query"] for m in self.memory]
            embeddings = self.encoder.encode(queries, convert_to_tensor=True)
            for i, mem in enumerate(self.memory):
                mem["_embedding"] = embeddings[i]


    def get_verified_contexts(self, current_query, threshold=0.85):
        """Finds similar historical queries and returns their successful context IDs."""
        if not self.memory:
            return []

        current_emb = self.encoder.encode(current_query, convert_to_tensor=True)

        verified_ids = []
        for mem in self.memory:
            sim = util.pytorch_cos_sim(current_emb, mem["_embedding"]).item()
            if sim > threshold:
                verified_ids.extend(mem["context_ids"])

        return list(set(verified_ids))

    @staticmethod
    def make_context_ids(contexts):
        return [make_chunk_id(context) for context in contexts]
