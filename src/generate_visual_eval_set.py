import os
import json
import random
import base64
import requests

METADATA_PATH = "data/frames_metadata.json"
EVAL_SET_PATH = "data/eval_set.json"
API_KEY = os.environ.get("GROQ_API_KEY")
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_visual_qa(image_path):
    base64_image = encode_image(image_path)
    
    prompt = """
    You are an expert professor. Look at this lecture slide/frame.
    Generate 1 specific question that requires seeing this image to answer.
    The answer should NOT be available in a typical audio transcript (e.g., ask about a specific diagram, a value in a tree, or a specific formula shown).
    
    OUTPUT FORMAT (JSON ONLY):
    {
        "question": "The question here",
        "ground_truth_answer": "The answer here"
    }
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=data)
        if response.status_code == 200:
            res_data = response.json()["choices"][0]["message"]["content"]
            # Extract JSON
            res_data = res_data.replace("```json", "").replace("```", "").strip()
            return json.loads(res_data)
    except Exception as e:
        print(f"Error for {image_path}: {e}")
    return None

def main():
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found.")
        return

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    # Pick 5 random frames to generate visual questions
    sampled_frames = random.sample(metadata, min(5, len(metadata)))
    
    print(f"Generating 5 Visual-Need questions...")
    visual_qa = []
    for i, frame in enumerate(sampled_frames):
        path = frame["path"]
        print(f"Processing frame {i+1}...")
        qa = generate_visual_qa(path)
        if qa:
            visual_qa.append(qa)
    
    # Append to existing eval set
    if os.path.exists(EVAL_SET_PATH):
        with open(EVAL_SET_PATH, "r") as f:
            existing = json.load(f)
    else:
        existing = []
        
    final_set = existing + visual_qa
    
    with open(EVAL_SET_PATH, "w") as f:
        json.dump(final_set, f, indent=4)
        
    print(f"Done! Added {len(visual_qa)} visual questions to {EVAL_SET_PATH}")

if __name__ == "__main__":
    main()
