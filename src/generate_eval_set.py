import os
import json
import random
from src.rag_chain import AgenticRAG

def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def generate_qa_pair(text_chunk, agent):
    prompt = f"""
    You are a Professor creating an exam for students based on the following lecture transcript segment.
    
    TRANSCRIPT SEGMENT:
    "{text_chunk}"
    
    TASK:
    Generate 1 specific question that can be answered ONLY using the information in this segment.
    Also provide the correct ground_truth_answer.
    
    OUTPUT FORMAT (JSON ONLY):
    {{
        "question": "The question here",
        "ground_truth_answer": "The answer here"
    }}
    """
    try:
        response = agent.groq_generate(prompt)
        # Clean up response to ensure valid JSON
        response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(response)
        return data
    except Exception as e:
        print(f"Error generating QA: {e}")
        return None

def main():
    transcript_path = "data/lecture_transcript.txt"
    output_path = "data/eval_set.json"
    
    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"Loading transcript from {transcript_path}...")
    text = load_transcript(transcript_path)
    chunks = chunk_text(text)
    
    # Randomly sample chunks to save time/cost, or use all
    selected_chunks = random.sample(chunks, min(10, len(chunks))) # Generate 10 QA pairs
    
    # Use absolute path to avoid issues, or relative if CWD is correct
    # data/vector_store.pkl exists based on list_dir
    vector_store_path = "data/vector_store.pkl"
    agent = AgenticRAG(vector_store_path) 
    
    eval_set = []
    print(f"Generating QA pairs from {len(selected_chunks)} chunks...")
    
    for i, chunk in enumerate(selected_chunks):
        qa = generate_qa_pair(chunk, agent)
        if qa:
            eval_set.append(qa)
            print(f"Generated pair {i+1}/{len(selected_chunks)}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=4)
    
    print(f"Saved {len(eval_set)} QA pairs to {output_path}")

if __name__ == "__main__":
    main()
