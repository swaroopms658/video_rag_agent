import os
from dotenv import load_dotenv

load_dotenv()

INITIAL_KEYS = [
    os.environ.get("GROQ_API_KEY_1", ""),
    os.environ.get("GROQ_API_KEY_2", ""),
    os.environ.get("GROQ_API_KEY", ""),  # Legacy single-key support
]

VALID_KEYS = [k for k in INITIAL_KEYS if k and k.strip()]

class KeyManager:
    def __init__(self):
        self.keys = VALID_KEYS
        self.current_index = 0
        if not self.keys:
            print("Warning: No API keys found in KeyManager!")

    def get_current_key(self):
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def rotate_key(self):
        """Switches to the next available key."""
        if not self.keys or len(self.keys) <= 1:
            print("Warning: Only 1 key available. Cannot rotate.")
            return self.get_current_key()
        
        old_key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        
        print(f"Rotating API Key: ...{old_key[-4:]} -> ...{new_key[-4:]}")
        return new_key
