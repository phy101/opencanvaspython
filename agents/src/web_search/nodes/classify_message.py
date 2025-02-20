from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from ..state import WebSearchState

class ClassificationSchema(BaseModel):
    should_search: bool = Field(
        ...,
        description="Whether to search the web based on the user's latest message."
    )

async def classify_message(state: WebSearchState) -> Dict[str, Any]:
    model = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0
    ).bind_tools(
        [ClassificationSchema],
        tool_choice={"type": "function", "function": {"name": "ClassificationSchema"}}
    )

    latest_message = state.messages[-1].content
    if isinstance(latest_message, list):
        latest_message = " ".join([item.text for item in latest_message if hasattr(item, "text")])
    
    response = await model.ainvoke([("user", latest_message)])
    
    if not response.tool_calls:
        return {"should_search": False}
    
    classification = response.tool_calls[0]["args"]
    return {"should_search": classification["should_search"]} 