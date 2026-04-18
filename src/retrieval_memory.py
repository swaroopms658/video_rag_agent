import json
import os
from sentence_transformers import SentenceTransformer, util

FEEDBACK_FILE = "data/rl_feedback.json"

class RetrievalMemory:
    def __init__(self):
        self.memory = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
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

        print(f"Loaded {len(self.memory)} verified memories.")

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
                print(f"   [Memory Hit] Query similar to: '{mem['query']}' (Sim: {sim:.2f})")

        return list(set(verified_ids))
