from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState
from ..shared.utils.urls import extract_urls
from .documents import (
    convert_context_document_to_human_message,
    fix_misformatted_context_doc_message
)
from ..utils import get_string_from_content
from .include_url_contents import include_url_contents
from .dynamic_determine_path import dynamic_determine_path

def extract_urls_from_last_message(messages: List[BaseMessage]) -> List[str]:
    recent_message = messages[-1]
    content = get_string_from_content(recent_message.content)
    return extract_urls(content)

async def generate_path(
    state: OpenCanvasGraphState,
    config: LangGraphRunnableConfig
) -> Dict[str, Any]:
    new_messages: List[BaseMessage] = []
    
    # Handle document messages
    doc_message = await convert_context_document_to_human_message(
        state._messages, config
    )
    existing_doc_message = next(
        (m for m in new_messages if any(
            c.type in ["document", "application/pdf"] 
            for c in getattr(m, 'content', [])
        )),
        None
    )
    
    if doc_message:
        new_messages.append(doc_message)
    elif existing_doc_message:
        fixed_messages = await fix_misformatted_context_doc_message(
            existing_doc_message, config
        )
        if fixed_messages:
            new_messages.extend(fixed_messages)

    # Check highlighted content
    if state.highlighted_code:
        return build_response("update_artifact", new_messages, state)
    if state.highlighted_text:
        return build_response("update_highlighted_text", new_messages, state)

    # Check rewrite themes
    if any([state.language, state.artifact_length, 
           state.regenerate_with_emojis, state.reading_level]):
        return build_response("rewrite_artifact_theme", new_messages, state)
    
    # Check code themes
    if any([state.add_comments, state.add_logs, 
           state.port_language, state.fix_bugs]):
        return build_response("rewrite_code_artifact_theme", new_messages, state)
    
    # Check custom actions
    if state.custom_quick_action_id:
        return build_response("custom_action", new_messages, state)
    
    # Check web search
    if state.web_search_enabled:
        return build_response("web_search", new_messages, state)

    # Handle URL content inclusion
    message_urls = extract_urls_from_last_message(state._messages)
    updated_message = None
    if message_urls:
        updated_message = await include_url_contents(
            state._messages[-1], message_urls
        )

    # Update message list
    updated_messages = state._messages.copy()
    if updated_message:
        updated_messages = [
            updated_message if msg.id == updated_message.id else msg 
            for msg in updated_messages
        ]

    # Determine path
    routing_result = await dynamic_determine_path({
        "state": state.copy(update={"_messages": updated_messages}),
        "new_messages": new_messages,
        "config": config
    })
    
    if not routing_result or not routing_result.get("route"):
        raise ValueError("Route not found")

    # Prepare final messages
    response = {"next": routing_result["route"]}
    if new_messages:
        response.update({
            "messages": new_messages,
            "_messages": updated_messages + new_messages
        })
    else:
        response["_messages"] = updated_messages

    return response

def build_response(
    next_node: str, 
    new_messages: List[BaseMessage], 
    state: OpenCanvasGraphState
) -> Dict[str, Any]:
    response = {"next": next_node}
    if new_messages:
        response.update({
            "messages": new_messages,
            "_messages": state._messages + new_messages
        })
    return response