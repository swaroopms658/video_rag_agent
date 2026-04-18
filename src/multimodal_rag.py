import os
import time
import pickle
import base64
import requests
from sentence_transformers import SentenceTransformer, util
from src.agent import SimpleRetriever
from src.key_manager import KeyManager

class VisualRetriever:
    def __init__(self, vector_store_path):
        with open(vector_store_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]

    def retrieve(self, query, model, top_k=1, boost_ids=None):
        query_emb = model.encode([query])
        sims = util.cos_sim(query_emb, self.embeddings)[0]
        
        # Apply Boosting from Memory
        if boost_ids:
            for i, chunk in enumerate(self.chunks):
                # Check if this chunk is in the boosted list
                # For images, chunk['path'] is the ID
                if chunk['path'] in boost_ids:
                    sims[i] += 0.2 # Boost score by 0.2 (significant)
        
        # Return top_k images with scores
        indices = sims.argsort(descending=True)[:top_k]
        return [(self.chunks[i], sims[i].item()) for i in indices]

class MultimodalRAG:
    def __init__(self, text_store_path, visual_store_path):
        print("Loading Text Retriever...")
        self.text_retriever = SimpleRetriever(text_store_path)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading Visual Retriever...")
        self.visual_retriever = VisualRetriever(visual_store_path)
        self.visual_model = SentenceTransformer('clip-ViT-B-32')
        
        self.key_manager = KeyManager()
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model_id = "meta-llama/llama-4-scout-17b-16e-instruct" # Vision-capable model

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def groq_generate_multimodal(self, prompt, image_paths):
        retries = 3
        while retries > 0:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                return "Error: No API Keys."

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare messages: Image FIRST, then Text
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add images first
            for path in image_paths:
                base64_image = self.encode_image(path)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            # Add text prompt second
            messages[0]["content"].append(
                {"type": "text", "text": prompt}
            )

            data = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.5
            }
            
            try:
                response = requests.post(self.endpoint, headers=headers, json=data)
                if response.status_code == 200:
                    completion = response.json()
                    return completion["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print(f"⚠️ Rate Limit Hit (429) on key ...{current_key[-4:]}. Rotating...")
                    wait_time = 30 * (4 - retries) # 30s, 60s, 90s
                    print(f"   Waiting {wait_time}s for quota cooldown...")
                    time.sleep(wait_time)
                    self.key_manager.rotate_key()
                    retries -= 1
                    continue
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    return "Error generating response."
            except Exception as e:
                print(f"Exception: {e}")
                return "Error during request."
        return "Failed after rate limit retries."

    def get_answer_with_context(self, query, boost_ids=None):
        # 1. Retrieve Text (with scores)
        text_items = self.text_retriever.retrieve(query, self.text_model, top_k=2)
        text_contexts = [item[0] for item in text_items]
        text_scores = [item[1] for item in text_items]
        
        text_context_str = "\n\n".join(text_contexts)
        
        # 2. Retrieve Visuals (with scores + boost)
        visual_items = self.visual_retriever.retrieve(query, self.visual_model, top_k=1, boost_ids=boost_ids)
        image_paths = [item[0]["path"] for item in visual_items]
        visual_scores = [item[1] for item in visual_items]
        
        # Calculate combined confidence
        all_scores = text_scores + visual_scores
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # 3. Construct Prompt
        prompt = f"""
        You are a helpful teaching assistant.
        Answer the student's question based on the provided Audio Context and the attached visual slides (images).
        
        AUDIO CONTEXT:
        {text_context_str}
        
        QUESTION:
        {query}
        
        INSTRUCTIONS:
        - Combine information from what was said (audio) and what is shown (visual).
        - If the visual contradicts the audio, trust the visual for formulas/diagrams.
        - Answer concisely.
        """
        
        print(f"Retrieving using {len(image_paths)} images...")
        answer = self.groq_generate_multimodal(prompt, image_paths)
        return answer, text_contexts, image_paths, avg_score

    def run(self, query):
        answer, _, _, _ = self.get_answer_with_context(query)
        return answer
