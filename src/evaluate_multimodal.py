from sentence_transformers import SentenceTransformer
from src.multimodal_rag import MultimodalRAG
from src.eval_utils import load_eval_set, calculate_similarity

EVAL_SET_PATH = "data/eval_set.json"
TEXT_STORE_PATH = "data/vector_store.pkl"
VISUAL_STORE_PATH = "data/visual_vector_store.pkl"

def run_evaluation():
    print("Starting Multimodal Evaluation...")
    
    # Load Data
    eval_set = load_eval_set(EVAL_SET_PATH)
    if not eval_set:
        print("No evaluation data found.")
        return

    # Init Models
    try:
        print(f"Initializing MultimodalRAG...")
        agent = MultimodalRAG(TEXT_STORE_PATH, VISUAL_STORE_PATH)
        print("Initializing Evaluation Model...")
        eval_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    total_sim = 0
    exact_match = 0
    count = 0

    print(f"\nEvaluating {len(eval_set)} QA pairs...")
    
    for item in eval_set:
        question = item.get("question")
        ground_truth = item.get("ground_truth_answer")
        
        if not question or not ground_truth:
            continue
            
        print(f"\nQ: {question}")
        try:
            # Generate Answer
            generated_answer = agent.run(question)
            print(f"A (Multimodal): {generated_answer}")
            print(f"A (GT): {ground_truth}")
            
            # Metrics
            sim = calculate_similarity(eval_model, generated_answer, ground_truth)
            total_sim += sim
            
            if generated_answer.strip().lower() == ground_truth.strip().lower():
                exact_match += 1
                
            print(f"Similarity: {sim:.4f}")
            count += 1
            
        except Exception as e:
            print(f"Error generating answer: {e}")

    if count > 0:
        avg_sim = total_sim / count
        em_score = exact_match / count
        print(f"\n--- Multimodal Results ---")
        print(f"Average Cosine Similarity: {avg_sim:.4f}")
        print(f"Exact Match Score: {em_score:.4f}")
    else:
        print("No questions evaluated.")

if __name__ == "__main__":
    run_evaluation()
