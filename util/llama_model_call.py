import os
import requests
from dotenv import load_dotenv
from graph.state import GraphState  

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS    = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def call_llm(prompt: str, max_new_tokens: int = 256):
    """
    Generic helper to call the Hugging Face Inference API for the LLaMA model.
    Returns the generated text (first choice).
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()
    # The API may return nested structure depending on model; adapt if needed.
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "generated_text" in result:
        return result["generated_text"]
    else:
        # Fallback: convert entire JSON to string
        return str(result)




