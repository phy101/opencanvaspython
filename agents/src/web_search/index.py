from langgraph.graph import StateGraph, END, START
from .state import WebSearchState
from .nodes import search, query_generator, classify_message

def search_or_end_conditional(state: WebSearchState) -> str:
    if state.should_search:
        return "query_generator"
    return END

# Initialize graph builder
builder = (
    StateGraph(WebSearchState)
    # Start node & edge
    .add_node("classify_message", classify_message)
    .add_node("query_generator", query_generator)
    .add_node("search", search)
    .add_edge(START, "classify_message")
    # Conditional edges
    .add_conditional_edges(
        "classify_message",
        search_or_end_conditional,
        ["query_generator", END]
    )
    # Edges
    .add_edge("query_generator", "search")
    .add_edge("search", END)
)

# Compile graph
graph = builder.compile()
graph.name = "Web Search Graph" 