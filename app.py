from flask import Flask, request, send_file
from graph.graph import app_graph
from graph.state import GraphState

app = Flask(__name__)

@app.route("/generate_audio", methods=["POST"])
def generate_audio():
    data = request.json
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400

    state = GraphState(text)
    outputs = app_graph.run(state)

    audio_path = state.get("audio_file")
    return send_file(audio_path, as_attachment=True, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)
