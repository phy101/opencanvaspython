from typing import Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState
from ..shared.types import ArtifactV3, ArtifactCodeV3
from ...utils import (
    create_context_document_messages,
    ensure_store_in_config,
    format_reflections,
    get_model_config,
    get_model_from_config,
    is_using_o1_mini_model
)
from ..prompts import UPDATE_HIGHLIGHTED_ARTIFACT_PROMPT

async def update_artifact(
    state: OpenCanvasGraphState,
    config: LangGraphRunnableConfig
) -> Dict[str, Any]:
    model_config = get_model_config(config)
    small_model = await get_model_from_config(config, {"temperature": 0})
    
    store = ensure_store_in_config(config)
    assistant_id = config.get("configurable", {}).get("assistant_id")
    if not assistant_id:
        raise ValueError("`assistant_id` not found in configurable")
    
    memory_namespace = ["memories", assistant_id]
    memory_key = "reflection"
    memories = await store.get(memory_namespace, memory_key)
    
    memories_str = format_reflections(
        memories.get("value") if memories else None
    ) if memories else "No reflections found."

    current_artifact_content = None
    if state.artifact and state.artifact.contents:
        current_artifact_content = state.artifact.contents[-1]
        if not isinstance(current_artifact_content, ArtifactCodeV3):
            raise ValueError("Current artifact content is not code")

    if not current_artifact_content or not state.highlighted_code:
        raise ValueError("Missing required artifact or highlight information")

    # Extract code sections
    start = max(0, state.highlighted_code.start_char_index - 500)
    end = min(len(current_artifact_content.code), state.highlighted_code.end_char_index + 500)
    
    before_highlight = current_artifact_content.code[start:state.highlighted_code.start_char_index]
    highlighted_text = current_artifact_content.code[
        state.highlighted_code.start_char_index:state.highlighted_code.end_char_index
    ]
    after_highlight = current_artifact_content.code[state.highlighted_code.end_char_index:end]

    formatted_prompt = UPDATE_HIGHLIGHTED_ARTIFACT_PROMPT.format(
        highlighted_text=highlighted_text,
        before_highlight=before_highlight,
        after_highlight=after_highlight,
        reflections=memories_str
    )

    recent_human_message = next(
        (msg for msg in reversed(state._messages) if msg.type == "human"),
        None
    )
    if not recent_human_message:
        raise ValueError("No recent human message found")

    context_docs = await create_context_document_messages(config)
    is_o1_mini = is_using_o1_mini_model(config)
    
    response = await small_model.invoke([
        {"role": "user" if is_o1_mini else "system", "content": formatted_prompt},
        *context_docs,
        recent_human_message
    ])

    # Rebuild the full code content
    full_code_before = current_artifact_content.code[:state.highlighted_code.start_char_index]
    full_code_after = current_artifact_content.code[state.highlighted_code.end_char_index:]
    updated_code = f"{full_code_before}{response.content}{full_code_after}"

    new_artifact_content = ArtifactCodeV3(
        index=len(state.artifact.contents) + 1,
        type="code",
        title=current_artifact_content.title,
        language=current_artifact_content.language,
        code=updated_code
    )

    new_artifact = ArtifactV3(
        current_index=len(state.artifact.contents) + 1,
        contents=[*state.artifact.contents, new_artifact_content]
    )

    return {"artifact": new_artifact} 