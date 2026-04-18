import os
import sys

# Ensure src in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

video_file = "data/7.mp4"

if not os.path.exists(video_file):
    print(f"Error: {video_file} not found.")
    sys.exit(1)

# 1. Transcribe
from src.transcribe import transcribe_audio
transcript_path = "data/transcript_7.txt"
if not os.path.exists(transcript_path):
    print("Transcribing 7.mp4 (this may take a few minutes)...")
    transcribe_audio(video_file, transcript_path)
else:
    print(f"Transcript exists at {transcript_path}")

# 2. Text Vector Store
from src.build_vectorstore import create_embeddings
text_store = "data/vector_store_7.pkl"
if not os.path.exists(text_store):
    print("Building Text Vector Store...")
    create_embeddings(transcript_path, text_store)
else:
    print(f"Text Vector store exists at {text_store}")

# 3. Extract Frames
import src.extract_frames as ef
ef.VIDEO_PATH = video_file
ef.OUTPUT_DIR = "data/frames_7"
ef.METADATA_PATH = "data/frames_metadata_7.json"
if not os.path.exists(ef.METADATA_PATH):
    print("Extracting frames...")
    ef.extract_frames()
else:
    print("Frames already extracted.")

# 4. Visual Vector Store
import src.generate_visual_embeddings as gve
gve.METADATA_PATH = ef.METADATA_PATH
gve.OUTPUT_PATH = "data/visual_vector_store_7.pkl"
if not os.path.exists(gve.OUTPUT_PATH):
    print("Generating visual embeddings...")
    gve.generate_embeddings()
else:
    print("Visual embeddings already exist.")

print("Processing complete!")
