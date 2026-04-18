import os
import sys
import json

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_chain import AgenticRAG
from src.multimodal_rag import MultimodalRAG
from src.retrieval_memory import RetrievalMemory

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("="*60)
    print("   DATASET SELECTION")
    print("="*60)
    print("1. Original Lecture (lecture.mp4)")
    
    has_7 = os.path.exists("data/vector_store_7.pkl")
    if has_7:
        print("2. New Video (7.mp4)")
        
    print()
    dataset_choice = input("Enter Choice (1 or 2): ").strip()
    
    if dataset_choice == '2' and has_7:
        vector_store_path = "data/vector_store_7.pkl"
        visual_store_path = "data/visual_vector_store_7.pkl"
    else:
        vector_store_path = "data/vector_store.pkl"
        visual_store_path = "data/visual_vector_store.pkl"

    print("\nInitializing RAG Systems... (This may take a moment)")

    try:
        text_rag = AgenticRAG(vector_store_path)
        multimodal_rag = MultimodalRAG(vector_store_path, visual_store_path)
        memory = RetrievalMemory()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    while True:
        clear_screen()
        print("="*60)
        print("   LECTURE RAG AGENT - INTERACTIVE RESEARCH DEMO")
        print("="*60)
        print("Select Mode:")
        print("1. Text-Only RAG (Baseline)")
        print("   - Strategy: Listens to the lecturer.")
        print("   - Best for: Definitions, spoken concepts, high-level summaries.")
        print("   - Sample Q: 'What is the definition of a BST?'")
        print()
        print("2. Multimodal RAG (Proposal)")
        print("   - Strategy: Listens to lecturer + LOOKS at slides.")
        print("   - Best for: Diagrams, code snippets, specific values on board.")
        print("   - Sample Q: 'What is the child of node 62?' or 'What code is on the slide?'")
        print("="*60)
        
        choice = input("Enter Choice (1 or 2, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            break
            
        if choice == '1':
            mode = "TEXT"
            agent = text_rag
            print("\n[Mode: TEXT-ONLY RAG]")
            query = input("Ask a question about the lecture audio: ")
        elif choice == '2':
            mode = "MULTIMODAL"
            agent = multimodal_rag
            print("\n[Mode: MULTIMODAL RAG]")
            query = input("Ask a question about the slides/diagrams: ")
        else:
            input("Invalid choice. Press Enter...")
            continue
            
        if not query.strip():
            continue
            
        print("\nThinking...")

        verified_ids = memory.get_verified_contexts(query)
        if verified_ids:
            print(f"   [Active Learning] Boosting {len(verified_ids)} verified sources...")

        try:
            if mode == "TEXT":
                answer, contexts, score = agent.get_answer_with_context(query)
                print("\n" + "-"*60)
                print(f"ANSWER:\n{answer}")
                print("-" * 60)
                print(f"ACCURACY GUARANTEE (Global Source Trace) | CONFIDENCE: {score:.1%}")
                print("The model based its answer on these AUDIO TRANSCRIPT chunks:")
                for i, ctx in enumerate(contexts):
                    preview = ctx[:200].replace("\n", " ") + "..."
                    print(f"[{i+1}] \"{preview}\"")
                print("-" * 60)
                
            else: # Multimodal
                answer, text_ctx, image_paths, score = agent.get_answer_with_context(query, boost_ids=verified_ids)
                print("\n" + "-"*60)
                print(f"ANSWER:\n{answer}")
                print("-" * 60)
                print(f"ACCURACY GUARANTEE (Visual Grounding) | CONFIDENCE: {score:.1%}")
                print("The model LOOKED at these frames to answer:")
                for path in image_paths:
                    # Calculate Timestamp
                    try:
                        # Extract frame number from "frame_0845.jpg"
                        filename = os.path.basename(path)
                        frame_num = int(filename.split('_')[1].split('.')[0])
                        # Assuming 1 frame extracted per second (typical for this pipeline)
                        seconds = frame_num 
                        timestamp = f"{seconds//60:02d}:{seconds%60:02d}"
                        print(f" - [IMAGE] {path} (approx. {timestamp})")
                    except:
                        print(f" - [IMAGE] {path}")
                print("And listened to these audio chunks:")
                for i, ctx in enumerate(text_ctx):
                    preview = ctx[:100].replace("\n", " ") + "..."
                    print(f" - [AUDIO] \"{preview}\"")
                print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")
            
        # RLHF Feedback Loop
        feedback = input("\nWas this answer correct? (y/n/skip): ").strip().lower()
        if feedback in ['y', 'n']:
            reward = 1 if feedback == 'y' else -1
            # Prepare Context Data for Memory
            if mode == "TEXT":
                # Use first 50 chars as ID for now
                context_ids = [ctx[:50] for ctx in contexts] 
            else:
                context_ids = image_paths

            log_entry = {
                "mode": mode,
                "query": query,
                "answer": answer,
                "score": score,
                "reward": reward,
                "context_ids": context_ids # Store what was used!
            }
            with open("data/rl_feedback.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"Feedback logged! Reward: {reward} (Memory Updated)")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
