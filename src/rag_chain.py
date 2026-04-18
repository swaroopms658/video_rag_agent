import os
import time
import requests
from sentence_transformers import SentenceTransformer
from src.agent import SimpleRetriever
from src.key_manager import KeyManager

class AgenticRAG:
    def __init__(self, vector_store_path):
        self.retriever = SimpleRetriever(vector_store_path)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.key_manager = KeyManager()
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def groq_generate(self, prompt):
        retries = 3
        while retries > 0:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise RuntimeError("No API Keys available.")

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.1-8b-instant", # Switched to 8B for speed/quota
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            try:
                response = requests.post(self.endpoint, headers=headers, json=data)
                
                if response.status_code == 200:
                    completion = response.json()
                    return completion["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print(f"⚠️ Rate Limit Hit (429) on key ...{current_key[-4:]}. Rotating...")
                    wait_time = 20 * (4 - retries) # 20s, 40s, 60s
                    print(f"   Waiting {wait_time}s for quota cooldown...")
                    time.sleep(wait_time)
                    self.key_manager.rotate_key()
                    retries -= 1
                    continue # Retry with new key
                else:
                    raise RuntimeError(f"Groq API error: {response.status_code} {response.text}")
            except Exception as e:
                print(f"Request Error: {e}")
                retries -= 1
        
        raise RuntimeError("Create Failed after retries (Rate Limits).")

    def get_answer_with_context(self, query):
        retrieved_items = self.retriever.retrieve(query, self.embed_model, top_k=3)
        contexts = [item[0] for item in retrieved_items]
        scores = [item[1] for item in retrieved_items]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        context_text = "\n\n".join(contexts)
        prompt = f"Answer the question based on the context below:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        answer = self.groq_generate(prompt)
        return answer, contexts, avg_score

    def run(self, query):
        answer, _, _ = self.get_answer_with_context(query)
        return answer
