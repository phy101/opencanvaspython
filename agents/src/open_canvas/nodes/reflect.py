import os
from typing import Dict, Any
from langchain.langgraph_sdk import Client
from langgraph.graph import LangGraphRunnableConfig
from ..state import OpenCanvasGraphState

async def reflect_node(
    state: OpenCanvasGraphState,
    config: LangGraphRunnableConfig
) -> Dict[str, Any]:
    try:
        lang_graph_client = Client(
            api_url=f"http://localhost:{os.getenv('PORT', '8000')}"
        )

        reflection_input = {
            "messages": state._messages,
            "artifact": state.artifact.dict() if state.artifact else None
        }
        
        reflection_config = {
            "configurable": {
                "open_canvas_assistant_id": config.get("configurable", {}).get("assistant_id")
            }
        }

        new_thread = await lang_graph_client.threads.create()
        
        await lang_graph_client.runs.create(
            thread_id=new_thread.thread_id,
            app_id="reflection",
            input=reflection_input,
            config=reflection_config,
            multitask_strategy="enqueue",
            after_seconds=5 * 60  # 5 minutes
        )
        
    except Exception as e:
        print(f"Failed to start reflection\n{e}")

    return {} 