"""Baseline retrieval systems for multi-system evaluation.

All baselines share the same abstract interface so evaluate.py can drive them
without any modification. Each concrete Retriever implements:
  - retrieve(query, model, top_k, boost_ids)  → [(chunk, score), ...]
  - retrieve_with_ids(query, model, top_k)    → [(chunk, score, chunk_id), ...]

BaselineAgent wraps a Retriever with AgenticRAG's generation logic so the
evaluate.py harness (which expects .retriever, .embed_model,
.get_answer_with_context, .groq_generate) works out of the box.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag_chain import AgenticRAG

CONFIDENCE_THRESHOLD = 0.35


class BaseRetriever(abc.ABC):
    """Common interface for all retrieval baselines."""

    @abc.abstractmethod
    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        """Returns [(chunk_text, score), ...] sorted by descending score."""

    @abc.abstractmethod
    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        """Returns [(chunk_text, score, chunk_id), ...] sorted by descending score."""


class BaselineAgent:
    """Pairs a custom BaseRetriever with an AgenticRAG for LLM generation.

    Drop-in replacement for AgenticRAG — exposes the same attributes that
    evaluate.py reads: .retriever, .embed_model, .groq_generate,
    .get_answer_with_context, .last_response_meta.
    """

    def __init__(self, name: str, retriever: BaseRetriever, rag_agent: "AgenticRAG"):
        self.name = name
        self.retriever = retriever
        self.embed_model = rag_agent.embed_model
        self._rag = rag_agent
        self.last_response_meta: dict = {
            "cache_hit": False,
            "estimated_llm_calls_saved": 0,
            "low_confidence": False,
        }

    def groq_generate(self, prompt: str) -> str:
        return self._rag.groq_generate(prompt)

    def get_answer_with_context(self, query: str, boost_ids=None, trace=None):
        items = self.retriever.retrieve(query, self.embed_model, top_k=3, boost_ids=boost_ids)
        contexts = [i[0] for i in items]
        scores = [i[1] for i in items]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score < CONFIDENCE_THRESHOLD:
            self.last_response_meta = {
                "cache_hit": False,
                "estimated_llm_calls_saved": 0,
                "low_confidence": True,
            }
            return (
                "I couldn't find relevant information in the lecture to answer confidently.",
                contexts,
                avg_score,
            )

        cached = self._rag.cache.lookup(query, contexts)
        if cached:
            self.last_response_meta = {
                "cache_hit": True,
                "estimated_llm_calls_saved": 1,
                "low_confidence": False,
            }
            return cached["answer"], contexts, avg_score

        context_text = "\n\n".join(contexts)
        prompt_text = (
            "You are a lecture assistant. Answer the question based only on the "
            "provided transcript context. Be concise and precise.\n\n"
            f"TRANSCRIPT CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n\nANSWER:"
        )
        answer = self.groq_generate(prompt_text)
        self._rag.cache.store(
            query, contexts, answer, metadata={"avg_retrieval_score": avg_score}
        )
        self.last_response_meta = {
            "cache_hit": False,
            "estimated_llm_calls_saved": 0,
            "low_confidence": False,
        }
        return answer, contexts, avg_score
