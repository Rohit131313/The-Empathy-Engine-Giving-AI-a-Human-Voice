import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base"

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def get_emotion(text: str):
    """
    Send text to Hugging Face API for emotion detection.
    Returns emotion label and confidence score.
    """
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    data = response.json()

    try:
        emotions = data[0]
        # Find highest confidence emotion
        top_emotion = max(emotions, key=lambda x: x['score'])
        return {
            "emotion": top_emotion["label"],
            "confidence": round(top_emotion["score"], 3)
        }
    except Exception as e:
        print("Error:", data)
        return {"error": str(e)}
