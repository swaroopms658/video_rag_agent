import json as _json
import os
import time
import requests
from sentence_transformers import SentenceTransformer
from src.agent import SimpleRetriever
from src.answer_cache import AnswerCache
from src.key_manager import KeyManager

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
HF_MODEL = os.getenv("HF_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")


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
        self._hf_client = None  # lazy init

    def _get_hf_client(self):
        if self._hf_client is None:
            hf_key = os.getenv("HF_API_KEY")
            if not hf_key:
                return None
            try:
                from huggingface_hub import InferenceClient
                self._hf_client = InferenceClient(provider="auto", api_key=hf_key, timeout=60)
            except ImportError:
                return None
        return self._hf_client

    def hf_generate(self, prompt, max_tokens=300):
        """Generate via HF InferenceClient (auto-routed provider). Used as Groq fallback."""
        client = self._get_hf_client()
        if client is None:
            raise RuntimeError("HF_API_KEY not set or huggingface_hub not installed.")
        result = client.chat_completion(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return result.choices[0].message.content

    def groq_generate(self, prompt):
        retries = 3
        while retries > 0:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                break

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3,
            }

            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=_json.dumps(data, ensure_ascii=False).encode("utf-8"),
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    wait_time = 5 * (4 - retries)
                    print(f"  [groq] 429 rate limit, waiting {wait_time}s ...")
                    time.sleep(wait_time)
                    self.key_manager.rotate_key()
                    retries -= 1
                    continue
                else:
                    raise RuntimeError(f"Groq API error: {response.status_code} {response.text}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"  [groq] request error: {e}")
                retries -= 1

        # Groq exhausted — fall back to HF
        print("  [groq] retries exhausted, falling back to HF ...")
        return self.hf_generate(prompt)

    def groq_try(self, prompt):
        """Single-attempt Groq call with short timeout; returns None on any failure."""
        key = self.key_manager.get_current_key()
        if not key:
            return None
        try:
            _body = _json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            }, ensure_ascii=False).encode("utf-8")
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                data=_body,
                timeout=8,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None

    def get_answer_with_context(self, query, boost_ids=None, trace=None):
        """ReAct (Reason + Act) loop for grounded lecture QA.

        Cycle:
          Thought 1 → Act: retrieve candidates
          Thought 2 → Act: check cost-aware cache
          Thought 3 → Act: generate + verify answer (or return cached)
        """
        def _t(step):
            if trace:
                trace(step)

        # ── Thought 1: What chunks are relevant to this query? ─────────────
        _t("thought:retrieve — selecting tool: FAISS retrieval")
        retrieved_items = self.retriever.retrieve(
            query,
            self.embed_model,
            top_k=3,
            boost_ids=boost_ids,
        )
        contexts = [item[0] for item in retrieved_items]
        scores   = [item[1] for item in retrieved_items]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # ── Observation 1: Is retrieval confidence sufficient? ─────────────
        if avg_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
            _t(f"observation:low_confidence (avg_score={avg_score:.3f} < {RETRIEVAL_CONFIDENCE_THRESHOLD})")
            self.last_response_meta = {
                "cache_hit": False, "estimated_llm_calls_saved": 0,
                "low_confidence": True,
                "react_steps": ["retrieve", "observe:low_confidence"],
            }
            return (
                "I couldn't find relevant information in the lecture transcript to answer this confidently. "
                "Try rephrasing or asking about a specific topic from the lecture.",
                contexts, avg_score,
            )

        # ── Thought 2: Has this query+context been answered before? ────────
        _t("thought:cache_check — selecting tool: answer cache lookup")
        cached = self.cache.lookup(query, contexts)
        if cached:
            _t("observation:cache_hit — returning cached answer (Objective 6: cost-aware caching)")
            self.last_response_meta = {
                "cache_hit": True, "estimated_llm_calls_saved": 1,
                "low_confidence": False,
                "react_steps": ["retrieve", "observe:confident", "cache_check", "observe:cache_hit"],
            }
            return cached["answer"], contexts, avg_score

        # ── Thought 3: Generate a grounded answer from transcript evidence ─
        _t("thought:generate — selecting tool: Groq LLM")
        context_text = "\n\n".join(contexts)
        prompt = f"""You are an agentic lecture assistant operating on retrieved transcript evidence.

Thought: I have retrieved {len(contexts)} relevant transcript chunks. I will synthesize
an answer supported only by this evidence and verify it is grounded before responding.

Retrieved transcript chunks:
{context_text}

Question: {query}

Act: Generate a concise, grounded answer using only the transcript evidence above.
If the context is insufficient, say so explicitly.

Answer:""".strip()

        _t("act:llm_call")
        answer = self.groq_generate(prompt)
        _t("observation:llm_done — storing to answer cache")

        self.cache.store(query, contexts, answer,
                         metadata={"avg_retrieval_score": avg_score})
        self.last_response_meta = {
            "cache_hit": False, "estimated_llm_calls_saved": 0,
            "low_confidence": False,
            "react_steps": ["retrieve", "observe:confident",
                            "cache_check", "observe:cache_miss",
                            "generate", "observe:answer_stored"],
        }
        return answer, contexts, avg_score

    def run(self, query):
        answer, _, _ = self.get_answer_with_context(query)
        return answer
