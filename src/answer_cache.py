import hashlib
import json
import os


class AnswerCache:
    def __init__(self, cache_path="data/answer_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load()

    def _load(self):
        if not os.path.exists(self.cache_path):
            return {}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, default=float)

    @staticmethod
    def _normalize_query(query):
        return " ".join(query.split()).strip().lower()

    def make_key(self, query, contexts):
        signature = {
            "query": self._normalize_query(query),
            "contexts": contexts,
        }
        payload = json.dumps(signature, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def lookup(self, query, contexts):
        key = self.make_key(query, contexts)
        entry = self.cache.get(key)
        if not entry:
            return None

        entry["hits"] = entry.get("hits", 0) + 1
        self.cache[key] = entry
        self._save()
        return entry

    def store(self, query, contexts, answer, metadata=None):
        key = self.make_key(query, contexts)
        self.cache[key] = {
            "query": self._normalize_query(query),
            "answer": answer,
            "contexts": contexts,
            "hits": 0,
            "metadata": metadata or {},
        }
        self._save()
