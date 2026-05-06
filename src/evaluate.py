"""Multi-system, multi-metric evaluation harness.

Usage (single system against current eval set):
    python -m src.evaluate

Usage (with explicit paths):
    python -m src.evaluate --eval data/lecture_rag_75/qa.jsonl \
                            --store data/lecture_rag_75 \
                            --system dense_minilm \
                            --output analysis/results

Each QA item in the eval set may optionally include:
    "gold_context_ids": ["<sha1_16>", ...]  — enables retrieval metrics (Hit@k, MRR, etc.)

If gold_context_ids are absent, only generation-quality metrics are computed.
"""

import argparse
import csv
import json
import os
import time

from sentence_transformers import SentenceTransformer

from src.rag_chain import AgenticRAG
from src.eval_utils import (
    load_eval_set,
    calculate_faithfulness,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    bertscore_f1,
    rouge_l,
    bleu_4,
)

EVAL_SET_PATH = "data/lecture_rag_75/qa.jsonl"
VECTOR_STORE_PATH = "data/lecture_rag_75/combined"
RESULTS_DIR = "analysis/results"
TOP_K_RETRIEVAL = 10   # for retrieval metrics
TOP_K_ANSWER = 3       # for answer generation


def evaluate_system(
    system_name,
    agent,
    eval_data,
    output_dir=RESULTS_DIR,
    top_k_retrieval=TOP_K_RETRIEVAL,
    top_k_answer=TOP_K_ANSWER,
    skip_bertscore=False,
    skip_llm_judge=False,
):
    """Evaluate one system against eval_data. Writes per-question CSV row.

    Returns a summary dict of macro-averaged metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{system_name}.csv")

    fieldnames = [
        "question", "gold_answer", "generated_answer", "retrieved_contexts",
        "hit_at_1", "hit_at_5", "mrr_score", "ndcg_at_10", "recall_at_10",
        "bertscore_f1", "rouge_l", "bleu_4", "faithfulness",
    ]
    metric_fields = [f for f in fieldnames if f not in
                     ("question", "gold_answer", "generated_answer", "retrieved_contexts")]
    accum = {k: [] for k in metric_fields}

    # Resume: load already-completed rows to skip re-processing
    done_questions: set[str] = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            done_questions = {r["question"] for r in existing if r.get("question")}
            for r in existing:
                for key in accum:
                    val = r.get(key, "")
                    if val and val != "None":
                        try:
                            accum[key].append(float(val))
                        except ValueError:
                            pass
            print(f"  [resume] {len(done_questions)} rows already done for {system_name}")
        except Exception as e:
            print(f"  [resume] could not read existing CSV: {e}")

    write_mode = "a" if done_questions else "w"
    with open(csv_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done_questions:
            writer.writeheader()

        for item in eval_data:
            question = item.get("question", "")
            gold_answer = item.get("ground_truth_answer", "")
            if question in done_questions:
                continue  # already processed
            gold_ids = item.get("gold_context_ids")  # may be None

            if not question or not gold_answer:
                continue

            # --- Retrieval (top-k for metrics) ---
            retrieval_results = agent.retriever.retrieve_with_ids(
                question, agent.embed_model, top_k=top_k_retrieval
            )
            retrieved_ids = [r[2] for r in retrieval_results]

            # --- Retrieval metrics (only when gold IDs available) ---
            if gold_ids:
                h1 = hit_at_k(retrieved_ids, gold_ids, 1)
                h5 = hit_at_k(retrieved_ids, gold_ids, 5)
                mrr_val = mrr(retrieved_ids, gold_ids)
                ndcg = ndcg_at_k(retrieved_ids, gold_ids, 10)
                rec = recall_at_k(retrieved_ids, gold_ids, 10)
            else:
                h1 = h5 = mrr_val = ndcg = rec = None

            # --- Answer generation (top-k_answer chunks) ---
            answer_contexts = [r[0] for r in retrieval_results[:top_k_answer]]
            try:
                generated_answer, _, _ = agent.get_answer_with_context(
                    question, boost_ids=None
                )
            except Exception as e:
                print(f"  Error generating answer: {e}")
                generated_answer = ""

            # --- Generation quality metrics ---
            bs = bertscore_f1([generated_answer], [gold_answer])[0] if not skip_bertscore else None
            rl = rouge_l(generated_answer, gold_answer)
            b4 = bleu_4(generated_answer, gold_answer)

            # --- Faithfulness (LLM judge) ---
            if not skip_llm_judge and generated_answer:
                faith = calculate_faithfulness(agent.groq_generate, generated_answer, answer_contexts)
            else:
                faith = None

            row = {
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "retrieved_contexts": json.dumps(answer_contexts, ensure_ascii=False),
                "hit_at_1": h1, "hit_at_5": h5, "mrr_score": mrr_val,
                "ndcg_at_10": ndcg, "recall_at_10": rec,
                "bertscore_f1": round(bs, 4) if bs is not None else None,
                "rouge_l": round(rl, 4),
                "bleu_4": round(b4, 4),
                "faithfulness": faith,
            }
            writer.writerow(row)
            f.flush()

            for key in accum:
                val = row[key]
                if val is not None:
                    accum[key].append(float(val))

            print(f"  [{system_name}] Q: {question[:60]}...")
            time.sleep(2)  # avoid Groq TPM limit
            mrr_s = f"{mrr_val:.3f}" if mrr_val is not None else "N/A"
            bs_s = f"{bs:.3f}" if bs is not None else "N/A"
            print(f"    R@1={h1}  R@5={h5}  MRR={mrr_s}  BS-F1={bs_s}  ROUGE-L={rl:.3f}")

    # Macro-average summary
    summary = {k: (sum(v) / len(v) if v else None) for k, v in accum.items()}
    summary["system"] = system_name
    summary["n_questions"] = sum(1 for v in accum["rouge_l"] if v is not None)
    summary["csv_path"] = csv_path

    print(f"\n=== {system_name} ({summary['n_questions']} questions) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    return summary


def run_evaluation(
    eval_path=EVAL_SET_PATH,
    vector_store_path=VECTOR_STORE_PATH,
    system_name="default",
    output_dir=RESULTS_DIR,
):
    """Convenience entry point; mirrors old API."""
    eval_data = load_eval_set(eval_path)
    if not eval_data:
        print("No evaluation data found.")
        return

    print(f"Initialising system '{system_name}' with store: {vector_store_path}")
    agent = AgenticRAG(vector_store_path)
    return evaluate_system(system_name, agent, eval_data, output_dir=output_dir)


def _build_agent_for_system(system_name: str, store_path: str,
                             rag_agent: "AgenticRAG", checkpoint: str = None):
    """Return an object with .retriever, .embed_model, .get_answer_with_context, .groq_generate."""
    from src.baselines import BaselineAgent
    from scripts.cold_start_eval import build_retriever  # reuse the factory

    retriever = build_retriever(
        system_name, store_path,
        checkpoint=checkpoint,
        checkpoint_cfrag="checkpoints/cfrag_lite",
    )
    return BaselineAgent(name=system_name, retriever=retriever, rag_agent=rag_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system(s)")
    parser.add_argument("--eval", default=EVAL_SET_PATH)
    parser.add_argument("--store", default=VECTOR_STORE_PATH)
    parser.add_argument("--systems", nargs="+",
                        default=["dense_minilm"],
                        help="System(s) to evaluate. One of: dense_minilm, dense_mpnet, "
                             "bm25, cross_encoder, static_memory, cfrag_lite, itma, "
                             "itma_cross, or 'default' for AgenticRAG.")
    parser.add_argument("--checkpoint", default=None,
                        help="ITMA / CFRAG-lite checkpoint path")
    parser.add_argument("--split", default="test",
                        choices=["all", "train", "dev", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--splits-file", default="data/lecture_rag_75/splits.json")
    parser.add_argument("--output", default=RESULTS_DIR)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    # Load eval data (optionally filtered by split)
    all_data = load_eval_set(args.eval)
    if args.split != "all" and os.path.exists(args.splits_file):
        import json
        with open(args.splits_file) as f:
            splits_data = json.load(f)
        allowed_ids = set(splits_data.get(args.split, []))
        eval_data = [it for it in all_data if it.get("id") in allowed_ids]
        print(f"Filtered to {args.split} split: {len(eval_data)}/{len(all_data)} items")
    else:
        eval_data = all_data

    if not eval_data:
        print("No evaluation data found.")
    else:
        # AgenticRAG is used for generation even when retriever is a baseline
        rag_agent = AgenticRAG(args.store)

        for sys_name in args.systems:
            print(f"\n{'='*60}\nSystem: {sys_name}\n{'='*60}")
            if sys_name == "default":
                agent = rag_agent
            else:
                agent = _build_agent_for_system(sys_name, args.store, rag_agent, args.checkpoint)

            evaluate_system(
                sys_name, agent, eval_data,
                output_dir=args.output,
                skip_bertscore=args.no_bertscore,
                skip_llm_judge=args.no_llm_judge,
            )
