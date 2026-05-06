import json
import math
from sentence_transformers import SentenceTransformer, util


def load_eval_set(path):
    """Load eval data from a JSON array or JSONL (one object per line)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                return json.load(f)
            # JSONL
            items = []
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            return items
    except Exception as e:
        print(f"Error loading eval set: {e}")
        return []


def calculate_similarity(model, text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def calculate_faithfulness(generate_fn, answer, contexts):
    """LLM-as-judge: 1.0 if answer is grounded in contexts, 0.0 otherwise."""
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


# ---------------------------------------------------------------------------
# Retrieval metrics  (work on ordered lists of chunk IDs)
# ---------------------------------------------------------------------------

def hit_at_k(retrieved_ids, relevant_ids, k):
    """1 if any relevant id appears in the top-k retrieved ids, else 0."""
    return int(bool(set(retrieved_ids[:k]) & set(relevant_ids)))


def mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank (single-query version — use mean over dataset)."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Normalised Discounted Cumulative Gain at k."""
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant ids found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


# ---------------------------------------------------------------------------
# Generation quality metrics
# ---------------------------------------------------------------------------

def bertscore_f1(predictions, references, lang="en"):
    """BERTScore F1 for parallel lists of predictions and references.

    Returns a list of floats, one per pair. Lazy-imports bert_score so the
    rest of the module is usable without it installed.
    """
    try:
        from bert_score import score as _bs
        _, _, F = _bs(predictions, references, lang=lang, verbose=False)
        return F.tolist()
    except ImportError:
        raise ImportError("pip install bert-score  to use bertscore_f1()")


def rouge_l(prediction, reference):
    """ROUGE-L F1 for a single prediction / reference pair."""
    try:
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    except ImportError:
        raise ImportError("pip install rouge-score  to use rouge_l()")


def bleu_4(prediction, reference):
    """Sentence-level BLEU-4, normalised to [0, 1]."""
    try:
        import sacrebleu as _sb
        result = _sb.sentence_bleu(prediction, [reference])
        return min(result.score / 100.0, 1.0)
    except ImportError:
        raise ImportError("pip install sacrebleu  to use bleu_4()")
