from typing import Optional, List
import uuid
from langchain_core.messages import HumanMessage
from ..utils import (
    FireCrawlLoader,
    get_model_from_config,
    traceable
)
from ..shared.types import SearchResult
from ..shared.constants import OC_WEB_SEARCH_RESULTS_MESSAGE_KEY

PROMPT = """You're an advanced AI assistant tasked with analyzing the user's message and determining if the user wants the contents of the URL included in their message included in their prompt.
You should ONLY answer 'true' if it is explicitly clear the user included the URL in their message so that its contents would be included in the prompt, otherwise, answer 'false'

Here is the user's message:
<message>
{message}
</message>

Now, given their message, determine whether or not they want the contents of that webpage to be included in the prompt."""

SCHEMA = {
    "name": "determine_include_url_contents",
    "description": "Whether to include URL contents in the prompt",
    "parameters": {
        "type": "object",
        "properties": {
            "should_include_url_contents": {
                "type": "boolean",
                "description": "Whether to include the URL contents in the prompt"
            }
        }
    }
}

@traceable(name="include_url_contents")
async def include_url_contents(
    message: HumanMessage,
    urls: List[str]
) -> Optional[HumanMessage]:
    try:
        prompt = PROMPT.format(message=message.content)
        
        # Initialize model
        model = await get_model_from_config(
            {"model_name": "gemini-2.0-flash", "model_provider": "google-genai"},
            {"temperature": 0}
        )
        model_with_tool = model.bind_tools([SCHEMA], tool_choice="determine_include_url_contents")
        
        # Get decision from model
        response = await model_with_tool.invoke([("user", prompt)])
        if not response.tool_calls:
            return None
            
        args = response.tool_calls[0]["args"]
        if not args.get("should_include_url_contents"):
            return None

        # Fetch URL contents
        contents = []
        for url in urls:
            loader = FireCrawlLoader(url=url, mode="scrape", params={"formats": ["markdown"]})
            docs = await loader.load()
            if docs:
                contents.append({
                    "url": url,
                    "content": docs[0].page_content
                })

        # Update message content
        updated_content = message.content
        for item in contents:
            updated_content = updated_content.replace(
                item["url"],
                f'<page-contents url="{item["url"]}">\n{item["content"]}\n</page-contents>'
            )

        return HumanMessage(
            id=f"web-content-{uuid.uuid4()}",
            content=updated_content,
            additional_kwargs={
                OC_WEB_SEARCH_RESULTS_MESSAGE_KEY: True,
                "webSearchResults": contents,
                "webSearchStatus": "done"
            }
        )

    except Exception as e:
        print(f"Error including URL contents: {e}")
        return None 