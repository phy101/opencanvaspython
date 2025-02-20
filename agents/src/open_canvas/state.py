from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from shared.src.types import (
    ArtifactLengthOptions,
    LanguageOptions,
    ProgrammingLanguageOptions,
    ReadingLevelOptions,
    CodeHighlight,
    ArtifactV3,
    TextHighlight,
    SearchResult
)
from shared.src.constants import OC_SUMMARIZED_MESSAGE_KEY

class OpenCanvasGraphState(BaseModel):
    """State representation for Open Canvas graph"""
    
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Full list of conversation messages"
    )
    _messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Internal messages list with summarization handling"
    )
    highlighted_code: Optional[CodeHighlight] = Field(
        default=None,
        description="Highlighted code section from artifact"
    )
    highlighted_text: Optional[TextHighlight] = Field(
        default=None,
        description="Highlighted text section from artifact"
    )
    artifact: Optional[ArtifactV3] = Field(
        default=None,
        description="Current generated artifact"
    )
    next: Optional[str] = Field(
        default=None,
        description="Next node to route to"
    )
    language: Optional[LanguageOptions] = Field(
        default=None,
        description="Target language for translations"
    )
    artifact_length: Optional[ArtifactLengthOptions] = Field(
        default=None,
        description="Target length for artifact generation"
    )
    regenerate_with_emojis: Optional[bool] = Field(
        default=None,
        description="Flag for emoji regeneration"
    )
    reading_level: Optional[ReadingLevelOptions] = Field(
        default=None,
        description="Target reading level"
    )
    add_comments: Optional[bool] = Field(
        default=None,
        description="Flag for adding code comments"
    )
    add_logs: Optional[bool] = Field(
        default=None,
        description="Flag for adding debug logs"
    )
    port_language: Optional[ProgrammingLanguageOptions] = Field(
        default=None,
        description="Target language for code porting"
    )
    fix_bugs: Optional[bool] = Field(
        default=None,
        description="Flag for bug fixing"
    )
    custom_quick_action_id: Optional[str] = Field(
        default=None,
        description="ID of custom quick action"
    )
    web_search_enabled: Optional[bool] = Field(
        default=None,
        description="Web search enable flag"
    )
    web_search_results: Optional[List[SearchResult]] = Field(
        default=None,
        description="Web search results"
    )

    def is_summary_message(self, msg: Union[BaseMessage, dict]) -> bool:
        """Check if message is a summary message"""
        if isinstance(msg, dict):
            return msg.get("kwargs", {}).get("additional_kwargs", {}).get(OC_SUMMARIZED_MESSAGE_KEY) is True
        return getattr(msg, "additional_kwargs", {}).get(OC_SUMMARIZED_MESSAGE_KEY) is True

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

OpenCanvasGraphReturnType = OpenCanvasGraphState 