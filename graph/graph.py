from langgraph.graph import END, StateGraph
from graph.consts import EMOTION_DETECT, EMOTION_DESC, RESPONSE_GEN, TTS_GEN
from graph.nodes import detect_emotion, generate_emotion_description, generate_response, generate_tts
from graph.state import GraphState

flow = StateGraph(state_schema=GraphState)

# Add nodes
flow.add_node(EMOTION_DETECT, detect_emotion)
flow.add_node(EMOTION_DESC, generate_emotion_description)
flow.add_node(RESPONSE_GEN, generate_response)
flow.add_node(TTS_GEN, generate_tts)

# Connect edges sequentially
flow.add_edge(EMOTION_DETECT, EMOTION_DESC)
flow.add_edge(EMOTION_DESC, RESPONSE_GEN)
flow.add_edge(RESPONSE_GEN, TTS_GEN)
flow.add_edge(TTS_GEN, END)

app_graph = flow.compile()
