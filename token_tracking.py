"""
Token usage tracking for Google GenAI LLM calls.

This module provides a callback handler to capture token usage from LLM responses,
since the synthesizer doesn't propagate raw LLM response data.
"""

from typing import Any, Dict, List, Optional
from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


class TokenUsageHandler(BaseCallbackHandler):
    """
    Callback handler to capture token usage from LLM calls.
    
    Google GenAI returns usage_metadata in the ChatResponse.raw dict,
    which includes prompt_token_count, candidates_token_count, and total_token_count.
    
    This handler captures that data from LLM events so it can be retrieved
    after synthesis operations.
    """
    
    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ) -> None:
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        self._usage_records: List[Dict[str, int]] = []
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
    
    def reset(self) -> None:
        """Reset all token counts."""
        self._usage_records = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
    
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Reset counts at the start of a new trace."""
        self.reset()
    
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass
    
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id
    
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Capture token usage from LLM events."""
        if event_type != CBEventType.LLM or payload is None:
            return
        
        # Try both 'response' and 'completion' keys (different LlamaIndex versions/methods)
        response = payload.get("response") or payload.get("completion")
        if response is None:
            return
        
        # Handle ChatResponse objects
        usage_data = self._extract_usage_from_response(response)
        if usage_data:
            self._usage_records.append(usage_data)
            self._total_prompt_tokens += usage_data.get("prompt_tokens", 0)
            self._total_completion_tokens += usage_data.get("completion_tokens", 0)
    
    def _extract_usage_from_response(self, response: Any) -> Optional[Dict[str, int]]:
        """
        Extract token usage from a response object.
        
        Handles multiple response formats:
        - ChatResponse with raw dict containing usage_metadata
        - ChatResponse with additional_kwargs containing token counts
        - Response objects with various usage formats
        """
        usage_data: Dict[str, int] = {}
        
        # Try to get from raw dict (Google GenAI format)
        if hasattr(response, "raw") and isinstance(response.raw, dict):
            raw = response.raw
            if "usage_metadata" in raw:
                metadata = raw["usage_metadata"]
                usage_data["prompt_tokens"] = metadata.get("prompt_token_count", 0)
                usage_data["completion_tokens"] = metadata.get("candidates_token_count", 0)
                usage_data["total_tokens"] = metadata.get("total_token_count", 0)
                return usage_data
        
        # Try to get from additional_kwargs (set by chat_from_gemini_response)
        if hasattr(response, "additional_kwargs") and isinstance(response.additional_kwargs, dict):
            kwargs = response.additional_kwargs
            if "prompt_tokens" in kwargs or "completion_tokens" in kwargs:
                usage_data["prompt_tokens"] = kwargs.get("prompt_tokens", 0)
                usage_data["completion_tokens"] = kwargs.get("completion_tokens", 0)
                usage_data["total_tokens"] = kwargs.get("total_tokens", 0)
                return usage_data
        
        # Try to get from message's additional_kwargs
        if hasattr(response, "message") and hasattr(response.message, "additional_kwargs"):
            kwargs = response.message.additional_kwargs
            if "prompt_tokens" in kwargs or "completion_tokens" in kwargs:
                usage_data["prompt_tokens"] = kwargs.get("prompt_tokens", 0)
                usage_data["completion_tokens"] = kwargs.get("completion_tokens", 0)
                usage_data["total_tokens"] = kwargs.get("total_tokens", 0)
                return usage_data
        
        return None if not usage_data else usage_data
    
    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all LLM calls."""
        return self._total_prompt_tokens
    
    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens across all LLM calls."""
        return self._total_completion_tokens
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all LLM calls."""
        return self._total_prompt_tokens + self._total_completion_tokens
    
    @property
    def usage_records(self) -> List[Dict[str, int]]:
        """List of usage records from individual LLM calls."""
        return self._usage_records.copy()
    
    def get_usage_summary(self) -> Dict[str, int]:
        """Get a summary of token usage."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_llm_calls": len(self._usage_records),
        }


def extract_token_usage_from_raw(raw: Any) -> Dict[str, int]:
    """
    Utility function to extract token usage from a raw response.
    
    This handles the different formats that Google GenAI may return:
    - Dictionary with usage_metadata key
    - Object with usage attribute
    
    Args:
        raw: The raw response from an LLM call
        
    Returns:
        Dict with prompt_tokens, completion_tokens keys (0 if not found)
    """
    result = {"prompt_tokens": 0, "completion_tokens": 0}
    
    if raw is None:
        return result
    
    # Handle dictionary format (Google GenAI via llama_index)
    if isinstance(raw, dict):
        usage = raw.get("usage_metadata", {})
        if usage:
            result["prompt_tokens"] = usage.get("prompt_token_count", 0)
            result["completion_tokens"] = usage.get("candidates_token_count", 0)
        return result
    
    # Handle object with usage attribute (OpenAI-style)
    if hasattr(raw, "usage") and raw.usage is not None:
        usage = raw.usage
        result["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
        result["completion_tokens"] = getattr(usage, "completion_tokens", 0)
        return result
    
    # Handle object with usage_metadata attribute
    if hasattr(raw, "usage_metadata") and raw.usage_metadata is not None:
        usage = raw.usage_metadata
        result["prompt_tokens"] = getattr(usage, "prompt_token_count", 0)
        result["completion_tokens"] = getattr(usage, "candidates_token_count", 0)
        return result
    
    return result
