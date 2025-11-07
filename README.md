# Empathy Engine

**Empathy Engine** is an AI-powered system that dynamically converts text into human-like speech, modulating the voice according to the detected emotion in the input text. It bridges the gap between text-based sentiment and expressive TTS (Text-to-Speech), generating responses with emotional resonance.  

The system uses a Hugging Face LLaMA model for both emotion-aware response generation and voice-style descriptions, combined with the **Parler-TTS** model for expressive audio output.  

---

## Project Structure

```

EmpathyEngine/
│
├─ graph/
│   ├─ consts.py           # Node identifiers
│   ├─ graph.py            # StateGraph definition & node connections
│   └─ nodes.py            # Node functions (emotion detection, response, TTS)
│
├─ util/
│   ├─ emotion_detector.py # Hugging Face API wrapper for emotion detection
│   └─ llama_model_llm.py  # Hugging Face API wrapper for LLaMA inference
│
├─ .env                    # Contains Hugging Face API key
├─ app.py                  # Flask backend entrypoint
└─ requirements.txt        # Python dependencies

````

---

## Features

1. **Text Input**: Accepts user input text via API (POST request).  
2. **Emotion Detection**: Classifies the text into emotions like Happy, Sad, Neutral, Angry using a Hugging Face model.  
3. **Emotion-to-Voice Mapping**: Uses LLaMA to generate a descriptive instruction for voice parameters like tone, pace, and expressiveness.  
4. **Emotion-Aware Response Generation**: Generates responses matching the detected emotion using the LLaMA model.  
5. **Text-to-Speech**: Converts the emotion-conditioned response into a WAV audio file using Parler-TTS.  
6. **State Tracking**: Maintains execution flow and state of the pipeline using `LangGraph StateGraph`.  

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Rohit131313/The-Empathy-Engine-Giving-AI-a-Human-Voice.git
cd The-Empathy-Engine-Giving-AI-a-Human-Voice
````

### 2. Create a Python virtual environment (recommended)

```bash
conda create --name voice
conda activate voice
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Set up environment variables

Create a `.env` file at the root of the project:

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### 5. Run the Flask server

```bash
python app.py
```

Server runs at `http://127.0.0.1:5000/`.

### 6. Test the API

Send a POST request to `/generate_audio` with JSON body:

```json
{
    "text": "I just got a promotion at work!"
}
```

* The server will return a **WAV audio file** with the response generated in a voice that matches the detected emotion.
* You can test via **Postman**, **curl**, or integrate into a frontend.

---

## Design Choices

1. **Emotion Detection**:

   * Hugging Face `j-hartmann/emotion-english-distilroberta-base` for multi-class emotion detection.
   * Captures emotions like Happy, Sad, Angry, and Neutral.

2. **Emotion-to-Voice Mapping**:

   * LLaMA (`meta-llama/Llama-3.1-8B-Instruct`) generates a **voice-style description** (tone, pace, expressiveness).
   * Example: `"Thomas speaks in a happy tone, moderate speed, expressive."`

3. **Response Generation**:

   * LLaMA generates responses conditioned on user input and detected emotion.
   * Ensures replies are emotionally consistent.

4. **Text-to-Speech (TTS)**:

   * **Parler-TTS mini-expresso** converts response text into audio.
   * Voice parameters are controlled indirectly through the LLaMA-generated emotion description.

5. **State Management**:

   * `LangGraph StateGraph` tracks the flow: Emotion Detection → Emotion Description → Response → TTS → End.
   * Logs stored in `state["flow"]` for debugging.

---

## Future Improvements / Stretch Goals

* Emotion intensity scaling: adjust TTS parameters based on confidence score.
* Expand emotion categories: include “surprised,” “concerned,” etc.
* Web UI with real-time text input and audio playback.
* SSML integration for advanced TTS control.
* Multiple voices for user personalization.


