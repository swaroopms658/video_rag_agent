import os
import time
import requests
from sentence_transformers import SentenceTransformer
from src.agent import SimpleRetriever
from src.answer_cache import AnswerCache
from src.key_manager import KeyManager

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35


class AgenticRAG:
    def __init__(self, vector_store_path):
        self.retriever = SimpleRetriever(vector_store_path)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.key_manager = KeyManager()
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.cache = AnswerCache()
        self.last_response_meta = {
            "cache_hit": False,
            "estimated_llm_calls_saved": 0,
            "low_confidence": False,
        }

    def groq_generate(self, prompt):
        retries = 3
        while retries > 0:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise RuntimeError("No API Keys available.")

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            try:
                response = requests.post(self.endpoint, headers=headers, json=data)
                
                if response.status_code == 200:
                    completion = response.json()
                    return completion["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print(f"⚠️ Rate Limit Hit (429) on key ...{current_key[-4:]}. Rotating...")
                    wait_time = 20 * (4 - retries) # 20s, 40s, 60s
                    print(f"   Waiting {wait_time}s for quota cooldown...")
                    time.sleep(wait_time)
                    self.key_manager.rotate_key()
                    retries -= 1
                    continue # Retry with new key
                else:
                    raise RuntimeError(f"Groq API error: {response.status_code} {response.text}")
            except Exception as e:
                print(f"Request Error: {e}")
                retries -= 1
        
        raise RuntimeError("Create Failed after retries (Rate Limits).")

    def groq_try(self, prompt):
        """Single-attempt Groq call with short timeout; returns None on any failure."""
        key = self.key_manager.get_current_key()
        if not key:
            return None
        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=8,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None

    def get_answer_with_context(self, query, boost_ids=None, trace=None):
        def _t(step):
            if trace:
                trace(step)

        _t("retrieving")
        retrieved_items = self.retriever.retrieve(
            query,
            self.embed_model,
            top_k=3,
            boost_ids=boost_ids,
        )
        contexts = [item[0] for item in retrieved_items]
        scores = [item[1] for item in retrieved_items]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
            self.last_response_meta = {
                "cache_hit": False,
                "estimated_llm_calls_saved": 0,
                "low_confidence": True,
            }
            return (
                "I couldn't find relevant information in the lecture transcript to answer this confidently. "
                "Try rephrasing or asking about a specific topic from the lecture.",
                contexts,
                avg_score,
            )

        _t("cache_check")
        cached = self.cache.lookup(query, contexts)
        if cached:
            _t("cache_hit")
            self.last_response_meta = {
                "cache_hit": True,
                "estimated_llm_calls_saved": 1,
                "low_confidence": False,
            }
            return cached["answer"], contexts, avg_score

        _t("cache_miss")
        context_text = "\n\n".join(contexts)
        prompt = f"""
You are an agentic lecture assistant operating on retrieved transcript evidence.

Follow this process:
1. Read the retrieved transcript chunks carefully.
2. Synthesize only the information supported by the transcript.
3. If the context is insufficient, say so explicitly instead of guessing.
4. Answer concisely and clearly.

TRANSCRIPT CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
""".strip()
        _t("llm_call")
        answer = self.groq_generate(prompt)
        _t("llm_done")
        self.cache.store(
            query,
            contexts,
            answer,
            metadata={"avg_retrieval_score": avg_score},
        )
        self.last_response_meta = {
            "cache_hit": False,
            "estimated_llm_calls_saved": 0,
            "low_confidence": False,
        }
        return answer, contexts, avg_score

    def run(self, query):
        answer, _, _ = self.get_answer_with_context(query)
        return answer
