from llama_index.core.bridge.pydantic import BaseModel
from typing import Optional


# Pydantic models for request and response

class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str
    chat_profile: str


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str
    chat_title: Optional[str] = None
    # tags_list: Optional[list] = None
    questions_list: Optional[list] = None
    time_taken: Optional[float] = None
    retrieved_sources: list[dict]
    retrieved_content: list[str]
    token_counts: Optional[dict] = None