from typing import Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState
from ..shared.types import ArtifactV3, ArtifactMarkdownV3
from ...utils import (
    create_context_document_messages,
    get_model_config,
    get_model_from_config,
    is_using_o1_mini_model
)

PROMPT = """You are an expert AI writing assistant, tasked with rewriting some text a user has selected. The selected text is nested inside a larger 'block'. You should always respond with ONLY the updated text block in accordance with the user's request.
You should always respond with the full markdown text block, as it will simply replace the existing block in the artifact.
The blocks will be joined later on, so you do not need to worry about the formatting of the blocks, only make sure you keep the formatting and structure of the block you are updating.

# Selected text
{highlighted_text}

# Text block
{text_blocks}

Your task is to rewrite the surrounding content to fulfill the users request. The selected text content you are provided above has had the markdown styling removed, so you can focus on the text itself.
However, ensure you ALWAYS respond with the full markdown text block, including any markdown syntax.
NEVER wrap your response in any additional markdown syntax, as this will be handled by the system. Do NOT include a triple backtick wrapping the text block unless it was present in the original text block.
You should NOT change anything EXCEPT the selected text. The ONLY instance where you may update the surrounding text is if it is necessary to make the selected text make sense.
You should ALWAYS respond with the full, updated text block, including any formatting, e.g newlines, indents, markdown syntax, etc. NEVER add extra syntax or formatting unless the user has specifically requested it.
If you observe partial markdown, this is OKAY because you are only updating a partial piece of the text.

Ensure you reply with the FULL text block including the updated selected text. NEVER include only the updated selected text, or additional prefixes or suffixes."""

async def update_highlighted_text(
    state: OpenCanvasGraphState,
    config: LangGraphRunnableConfig
) -> Dict[str, Any]:
    model_config = get_model_config(config)
    model = await get_model_from_config(config, {"temperature": 0})
    
    current_artifact_content = None
    if state.artifact and state.artifact.contents:
        current_artifact_content = state.artifact.contents[-1]
        if not isinstance(current_artifact_content, ArtifactMarkdownV3):
            raise ValueError("Artifact is not markdown content")

    if not state.highlighted_text or not current_artifact_content:
        raise ValueError("Missing required highlight information or artifact")

    formatted_prompt = PROMPT.format(
        highlighted_text=state.highlighted_text.selected_text,
        text_blocks=state.highlighted_text.markdown_block
    )

    recent_user_message = state._messages[-1] if state._messages else None
    if not recent_user_message or recent_user_message.type != "human":
        raise ValueError("Expected a human message")

    context_docs = await create_context_document_messages(config)
    is_o1_mini = is_using_o1_mini_model(config)
    
    response = await model.invoke([
        {"role": "user" if is_o1_mini else "system", "content": formatted_prompt},
        *context_docs,
        recent_user_message
    ])

    # Rebuild the full markdown content
    full_markdown = current_artifact_content.full_markdown.replace(
        state.highlighted_text.markdown_block,
        response.content
    )

    new_artifact_content = ArtifactMarkdownV3(
        index=len(state.artifact.contents) + 1,
        type="text",
        title=current_artifact_content.title,
        full_markdown=full_markdown
    )

    new_artifact = ArtifactV3(
        current_index=len(state.artifact.contents) + 1,
        contents=[*state.artifact.contents, new_artifact_content]
    )

    return {"artifact": new_artifact} 