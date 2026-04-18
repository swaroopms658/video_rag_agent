import json
import os
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

METADATA_PATH = "data/frames_metadata.json"
OUTPUT_PATH = "data/visual_vector_store.pkl"
MODEL_NAME = "clip-ViT-B-32"

def generate_embeddings():
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}. Run extract_frames.py first.")
        return

    print(f"Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loading CLIP model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    images = []
    valid_metadata = []
    
    print("Loading images...")
    for item in metadata:
        path = item["path"]
        if os.path.exists(path):
            try:
                img = Image.open(path)
                images.append(img)
                valid_metadata.append(item)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        else:
            print(f"Image missing: {path}")
    
    if not images:
        print("No images to embed.")
        return

    print(f"Generating embeddings for {len(images)} images...")
    embeddings = model.encode(images, batch_size=32, show_progress_bar=True)
    
    print(f"Saving visual vector store to {OUTPUT_PATH}...")
    data = {
        "chunks": valid_metadata, # We use metadata as "chunks" for visual retrieval
        "embeddings": embeddings
    }
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)
        
    print("Done!")

if __name__ == "__main__":
    generate_embeddings()
