from typing import Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState
from ..shared.types import ArtifactV3
from ...utils import (
    create_context_document_messages,
    get_formatted_reflections,
    get_model_config,
    get_model_from_config,
    is_using_o1_mini_model,
    optionally_get_system_prompt_from_config
)
from .schemas import ARTIFACT_TOOL_SCHEMA
from .utils import create_artifact_content, format_new_artifact_prompt

async def generate_artifact(
    state: OpenCanvasGraphState,
    config: LangGraphRunnableConfig
) -> Dict[str, Any]:
    model_config = get_model_config(config, is_tool_calling=True)
    model_name = model_config.get("model_name")
    
    small_model = await get_model_from_config(
        config,
        {"temperature": 0.5, "is_tool_calling": True}
    )
    
    # Bind tool to model
    model_with_tool = small_model.bind_tools(
        [{
            "name": "generate_artifact",
            "description": ARTIFACT_TOOL_SCHEMA["description"],
            "schema": ARTIFACT_TOOL_SCHEMA
        }],
        tool_choice="generate_artifact"
    )

    memories_str = await get_formatted_reflections(config)
    formatted_prompt = format_new_artifact_prompt(memories_str, model_name)

    user_prompt = optionally_get_system_prompt_from_config(config)
    full_prompt = f"{user_prompt}\n{formatted_prompt}" if user_prompt else formatted_prompt

    context_docs = await create_context_document_messages(config)
    is_o1_mini = is_using_o1_mini_model(config)

    response = await model_with_tool.invoke(
        [
            {"role": "user" if is_o1_mini else "system", "content": full_prompt},
            *context_docs,
            *state._messages
        ],
        {"run_name": "generate_artifact"}
    )

    if not response.tool_calls or not response.tool_calls[0].args:
        raise ValueError("No valid tool arguments found in response")

    new_content = create_artifact_content(response.tool_calls[0].args)
    
    new_artifact = ArtifactV3(
        current_index=1,
        contents=[new_content]
    )

    return {"artifact": new_artifact} 