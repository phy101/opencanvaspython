from typing import Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState
from ..prompts import (
    ROUTE_QUERY_PROMPT,
    ROUTE_QUERY_OPTIONS_HAS_ARTIFACTS,
    ROUTE_QUERY_OPTIONS_NO_ARTIFACTS,
    CURRENT_ARTIFACT_PROMPT,
    NO_ARTIFACT_PROMPT
)
from ..utils import (
    format_artifact_content_with_template,
    get_model_from_config,
    create_context_document_messages
)
from ..shared.utils.artifacts import get_artifact_content
from langsmith import traceable

class RouteSchema(BaseModel):
    route: str = Field(..., description="The route to take based on the user's query.")

async def dynamic_determine_path(
    state: OpenCanvasGraphState,
    new_messages: List[BaseMessage],
    config: LangGraphRunnableConfig
) -> Optional[Dict[str, Any]]:
    current_artifact_content = None
    if state.artifact:
        current_artifact_content = get_artifact_content(state.artifact)

    # Format the prompt
    artifact_options = ROUTE_QUERY_OPTIONS_HAS_ARTIFACTS if current_artifact_content else ROUTE_QUERY_OPTIONS_NO_ARTIFACTS
    recent_messages = "\n\n".join(
        f"{msg.type}: {msg.content}" for msg in state._messages[-3:]
    )
    current_artifact_prompt = format_artifact_content_with_template(
        CURRENT_ARTIFACT_PROMPT, current_artifact_content
    ) if current_artifact_content else NO_ARTIFACT_PROMPT

    formatted_prompt = ROUTE_QUERY_PROMPT.format(
        artifact_options=artifact_options,
        recent_messages=recent_messages,
        current_artifact_prompt=current_artifact_prompt
    )

    # Determine possible routes
    artifact_route = "rewrite_artifact" if current_artifact_content else "generate_artifact"
    
    # Prepare model
    model = await get_model_from_config(config, {"temperature": 0, "is_tool_calling": True})
    model_with_tool = model.bind_tools(
        [{
            "name": "route_query",
            "description": "The route to take based on the user's query.",
            "schema": RouteSchema.schema()
        }],
        tool_choice="route_query"
    )

    # Get context documents
    context_docs = await create_context_document_messages(config)
    
    # Prepare input messages
    messages = [
        *context_docs,
        *new_messages,
        {"role": "user", "content": formatted_prompt}
    ]

    # Invoke model
    response = await model_with_tool.invoke(messages)
    
    if response.tool_calls and len(response.tool_calls) > 0:
        return response.tool_calls[0].args
    return None 