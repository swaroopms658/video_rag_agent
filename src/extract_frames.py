import cv2
import os
import json
import math

VIDEO_PATH = "data/lecture.mp4"
OUTPUT_DIR = "data/frames"
METADATA_PATH = "data/frames_metadata.json"
INTERVAL_SECONDS = 5  # Extract one frame every 5 seconds

def extract_frames():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps}, Duration: {duration:.2f}s")
    
    frame_interval = int(fps * INTERVAL_SECONDS)
    metadata = []
    
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            timestamp = count / fps
            frame_filename = f"frame_{int(timestamp):04d}.jpg"
            frame_path = os.path.join(OUTPUT_DIR, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Save metadata
            metadata.append({
                "frame_id": saved_count,
                "timestamp": timestamp,
                "path": frame_path,
                "filename": frame_filename
            })
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...")
        
        count += 1
    
    cap.release()
    
    # Save metadata to JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Extraction complete. Saved {saved_count} frames to {OUTPUT_DIR}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    extract_frames()
