from langgraph.graph import StateGraph, END, START, Send, Command
from langchain_core.messages import BaseMessage
from shared.src.constants import DEFAULT_INPUTS
from .nodes import (
    custom_action,
    generate_artifact,
    generate_followup,
    generate_path,
    reflect_node,
    rewrite_artifact,
    rewrite_artifact_theme,
    update_artifact,
    reply_to_general_input,
    rewrite_code_artifact_theme,
    generate_title_node,
    update_highlighted_text,
    summarizer
)
from .state import OpenCanvasGraphState
from ..web_search.index import graph as web_search_graph
from ..utils import create_ai_message_from_web_results

def route_node(state: OpenCanvasGraphState) -> Send:
    if not state.next:
        raise ValueError("'next' state field not set")
    return Send(state.next, state.copy(update=state.dict()))

def clean_state(_: OpenCanvasGraphState) -> dict:
    return DEFAULT_INPUTS.copy()

CHARACTER_MAX = 300000

def simple_token_calculator(state: OpenCanvasGraphState) -> str:
    total_chars = 0
    for msg in state._messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        else:
            total_chars += sum(len(c.text) for c in msg.content if hasattr(c, 'text'))
    return "summarizer" if total_chars > CHARACTER_MAX else END

def conditionally_generate_title(state: OpenCanvasGraphState) -> str:
    if len(state.messages) > 2:
        return simple_token_calculator(state)
    return "generate_title"

def route_post_web_search(state: OpenCanvasGraphState) -> Command:
    has_artifacts = len(state.artifact.contents) > 1 if state.artifact else False
    if not state.web_search_results:
        target = "rewrite_artifact" if has_artifacts else "generate_artifact"
        return Send(target, state.copy(update={"web_search_enabled": False}))
    
    web_search_msg = create_ai_message_from_web_results(state.web_search_results)
    return Command(
        goto="rewrite_artifact" if has_artifacts else "generate_artifact",
        update={
            "web_search_enabled": False,
            "messages": [web_search_msg],
            "_messages": [web_search_msg]
        }
    )

# Initialize graph builder
builder = (
    StateGraph(OpenCanvasGraphState)
    # Start node & edge
    .add_node("generatePath", generate_path)
    .add_edge(START, "generatePath")
    # Nodes
    .add_node("replyToGeneralInput", reply_to_general_input)
    .add_node("rewriteArtifact", rewrite_artifact)
    .add_node("rewriteArtifactTheme", rewrite_artifact_theme)
    .add_node("rewriteCodeArtifactTheme", rewrite_code_artifact_theme)
    .add_node("updateArtifact", update_artifact)
    .add_node("updateHighlightedText", update_highlighted_text)
    .add_node("generateArtifact", generate_artifact)
    .add_node("customAction", custom_action)
    .add_node("generateFollowup", generate_followup)
    .add_node("cleanState", clean_state)
    .add_node("reflect", reflect_node)
    .add_node("generateTitle", generate_title_node)
    .add_node("summarizer", summarizer)
    .add_node("webSearch", web_search_graph)
    .add_node("routePostWebSearch", route_post_web_search)
    # Initial router
    .add_conditional_edges(
        "generatePath",
        route_node,
        [
            "updateArtifact",
            "rewriteArtifactTheme",
            "rewriteCodeArtifactTheme",
            "replyToGeneralInput",
            "generateArtifact",
            "rewriteArtifact",
            "customAction",
            "updateHighlightedText",
            "webSearch",
        ]
    )
    # Edges
    .add_edge("generateArtifact", "generateFollowup")
    .add_edge("updateArtifact", "generateFollowup")
    .add_edge("updateHighlightedText", "generateFollowup")
    .add_edge("rewriteArtifact", "generateFollowup")
    .add_edge("rewriteArtifactTheme", "generateFollowup")
    .add_edge("rewriteCodeArtifactTheme", "generateFollowup")
    .add_edge("customAction", "generateFollowup")
    .add_edge("webSearch", "routePostWebSearch")
    # End edges
    .add_edge("replyToGeneralInput", "cleanState")
    # Only reflect if an artifact was generated/updated
    .add_edge("generateFollowup", "reflect")
    .add_edge("reflect", "cleanState")
    .add_conditional_edges(
        "cleanState",
        conditionally_generate_title,
        [END, "generateTitle", "summarizer"]
    )
    .add_edge("generateTitle", END)
    .add_edge("summarizer", END)
)

# Compile graph
graph = builder.compile().configure(run_name="open_canvas") 