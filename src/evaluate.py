from sentence_transformers import SentenceTransformer
from src.rag_chain import AgenticRAG
from src.eval_utils import load_eval_set, calculate_similarity, calculate_faithfulness

EVAL_SET_PATH = "data/eval_set.json"
VECTOR_STORE_PATH = "data/vector_store.pkl"

def run_evaluation():
    print("Starting Evaluation...")

    eval_set = load_eval_set(EVAL_SET_PATH)
    if not eval_set:
        print("No evaluation data found. Run generate_eval_set.py first.")
        return

    try:
        print(f"Initializing AgenticRAG with {VECTOR_STORE_PATH}...")
        agent = AgenticRAG(VECTOR_STORE_PATH)
        print("Initializing Evaluation Model...")
        eval_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    total_sim = 0
    exact_match = 0
    total_faithfulness = 0
    faithfulness_count = 0
    count = 0

    print(f"\nEvaluating {len(eval_set)} QA pairs...")

    for item in eval_set:
        question = item.get("question")
        ground_truth = item.get("ground_truth_answer")

        if not question or not ground_truth:
            continue

        print(f"\nQ: {question}")
        try:
            generated_answer, contexts, _ = agent.get_answer_with_context(question)
            print(f"A (RAG): {generated_answer}")
            print(f"A (GT):  {ground_truth}")

            sim = calculate_similarity(eval_model, generated_answer, ground_truth)
            total_sim += sim

            if generated_answer.strip().lower() == ground_truth.strip().lower():
                exact_match += 1

            faith = calculate_faithfulness(agent.groq_generate, generated_answer, contexts)
            if faith is not None:
                total_faithfulness += faith
                faithfulness_count += 1
                faith_label = "FAITHFUL" if faith == 1.0 else "HALLUCINATION"
                print(f"Similarity: {sim:.4f}  |  Faithfulness: {faith_label}")
            else:
                print(f"Similarity: {sim:.4f}  |  Faithfulness: (error)")

            count += 1

        except Exception as e:
            print(f"Error generating answer: {e}")

    if count > 0:
        avg_sim = total_sim / count
        em_score = exact_match / count
        print(f"\n--- Results ({count} questions) ---")
        print(f"Average Cosine Similarity : {avg_sim:.4f}")
        print(f"Exact Match Score         : {em_score:.4f}")
        if faithfulness_count > 0:
            avg_faith = total_faithfulness / faithfulness_count
            print(f"Faithfulness Score        : {avg_faith:.4f}  ({faithfulness_count}/{count} judged)")
    else:
        print("No questions evaluated.")

if __name__ == "__main__":
    run_evaluation()
