import json
from sentence_transformers import SentenceTransformer, util


def load_eval_set(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading eval set: {e}")
        return []


def calculate_similarity(model, text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def calculate_faithfulness(generate_fn, answer, contexts):
    """
    LLM-as-judge: returns 1.0 if every claim in the answer is supported by
    the retrieved transcript chunks, 0.0 otherwise.
    """
    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""You are a strict fact-checker for a lecture Q&A system.

RETRIEVED TRANSCRIPT CHUNKS:
\"\"\"{context_text}\"\"\"

GENERATED ANSWER:
\"\"\"{answer}\"\"\"

Task: Does the generated answer contain ONLY information that is explicitly stated or directly implied by the transcript chunks above?
- Answer YES if every claim in the answer is grounded in the transcript.
- Answer NO if the answer contains any claim not found in the transcript (i.e. hallucination).

Respond with a single word: YES or NO."""
    try:
        response = generate_fn(prompt).strip().upper()
        return 1.0 if response.startswith("YES") else 0.0
    except Exception as e:
        print(f"Error in faithfulness check: {e}")
        return None
