from util.emotion_detector import get_emotion
from util.llama_model_call import call_llm
import torch
from graph.state import GraphState
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf


# -------------------
# 1️⃣ Emotion Detection Node
# -------------------
def detect_emotion(state:GraphState):
    text = state["text"]
    result = get_emotion(text)
    emotion = result.get("emotion", "neutral")
    state["emotion"] = emotion
    state["flow"].append(f"Detected emotion: {emotion}")
    return state

# -------------------
# 2️⃣ Emotion Description Generator Node
# -------------------
def generate_emotion_description(state: GraphState):
    emotion = state.get("emotion", "neutral")
    prompt = (
        f"You are a voice‑style designer. "
        f"Given the detected emotion: '{emotion}', "
        f"generate a concise descriptive instruction "
        f"for how the voice should sound (tone, pace, expressiveness) "
        f"for TTS generation."
    )
    desc = call_llm(prompt, max_new_tokens=50).strip()
    state["emotion_description"] = desc
    state["flow"].append(f"Generated emotion description: {desc}")
    return state

# -------------------
# 3️⃣ Response Generator Node
# -------------------
def generate_response(state: GraphState):
    text = state["text"]
    emotion = state.get("emotion", "neutral")
    prompt = (
        f"You are a conversational agent. The user said: \"{text}\". "
        f"The detected emotion is '{emotion}'. "
        f"Generate a suitable response that reflects this emotion in tone, style and word‑choice."
    )
    response = call_llm(prompt, max_new_tokens=150).strip()
    state["response_text"] = response
    state["flow"].append(f"Generated response: {response}")
    return state

# -------------------
# 4️⃣ TTS Node
# -------------------
def generate_tts(state:GraphState):
    response_text = state.get("response_text", "")
    emotion_desc = state.get("emotion_description", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-expresso"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

    input_ids = tokenizer(emotion_desc, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(response_text, return_tensors="pt").input_ids.to(device)

    audio_arr = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).cpu().numpy().squeeze()
    file_path = "output.wav"
    sf.write(file_path, audio_arr, model.config.sampling_rate)
    state["audio_file"] = file_path
    state["flow"].append(f"TTS audio generated: {file_path}")
    return state
