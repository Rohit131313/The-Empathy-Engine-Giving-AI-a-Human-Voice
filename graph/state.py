class GraphState(dict):
    def __init__(self, text):
        super().__init__()
        self["text"] = text
        self["flow"] = []