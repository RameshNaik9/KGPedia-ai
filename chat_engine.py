from typing import Any, List, Optional

import time

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.base.response.schema import (
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.tools.types import ToolOutput
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.chat_engine.utils import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from utils import measure_time
from cache import NodeCache

def get_prefix_messages_with_context(
    context_template: PromptTemplate,
    system_prompt: str,
    prefix_messages: List[ChatMessage],
    chat_history: List[ChatMessage],
    llm_metadata_system_role: MessageRole,
) -> List[ChatMessage]:
    context_str_w_sys_prompt = str(context_template) + system_prompt.strip()
    return [
        ChatMessage(content=context_str_w_sys_prompt, role=llm_metadata_system_role),
        *prefix_messages,
        *chat_history,
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]

DEFAULT_CONTEXT_TEMPLATE = (
    "Use the context information below to assist the user."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

DEFAULT_REFINE_TEMPLATE = (
    "Using the context below, refine the following existing answer using the provided context to assist the user.\n"
    "If the context isn't helpful, just repeat the existing answer and nothing more.\n"
    "\n--------------------\n"
    "{context_msg}"
    "\n--------------------\n"
    "Existing Answer:\n"
    "{existing_answer}"
    "\n--------------------\n"
)

class ContextChatEngine(BaseChatEngine):
    """
    Context Chat Engine.

    Uses a retriever to retrieve a context, set the context in the system prompt,
    and then uses an LLM to generate a response, for a fluid chat experience.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        context_refine_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE
        self._context_refine_template = (
            context_refine_template or DEFAULT_REFINE_TEMPLATE
        )

        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager
        self._node_cache = NodeCache()

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        context_refine_template: Optional[str] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "ContextChatEngine":
        """Initialize a ContextChatEngine from default parameters."""
        llm = llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=Settings.callback_manager,
            context_template=context_template,
            context_refine_template=context_refine_template,
        )

    @measure_time("Getting nodes from retriever")
    def _get_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        # Check cache first
        cached_nodes = self._node_cache.get(message)
        if cached_nodes is not None:
            print("✨ Cache hit! Using cached nodes")
            return cached_nodes

        # If not in cache, retrieve normally
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )
        
        # Store in cache
        self._node_cache.put(message, nodes)
        return nodes

    async def _aget_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes   

    # @measure_time("Getting response synthesizer")
    def _get_response_synthesizer(
        self, chat_history: List[ChatMessage], streaming: bool = False
    ) -> CompactAndRefine:
        # Pull the system prompt from the prefix messages
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]

        # Get the messages for the QA and refine prompts
        qa_messages = get_prefix_messages_with_context(
            PromptTemplate(self._context_template),
            system_prompt,
            prefix_messages,
            chat_history,
            self._llm.metadata.system_role,
        )
        refine_messages = get_prefix_messages_with_context(
            PromptTemplate(self._context_refine_template),
            system_prompt,
            prefix_messages,
            chat_history,
            self._llm.metadata.system_role,
        )

        # Get the response synthesizer
        return get_response_synthesizer(
            self._llm, self.callback_manager, qa_messages, refine_messages, streaming
        )

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # Record the Unix timestamp for when the user sends the message
        user_timestamp = int(time.time())
        nodes = self._get_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        chat_history = self._memory.get(input=message)
        synthesizer = self._get_response_synthesizer(chat_history)

        # Generate the assistant's response
        response = synthesizer.synthesize(message, nodes)

        assistant_timestamp = int(time.time())

        user_message = ChatMessage(
            content=message,
            role=MessageRole.USER,
            additional_kwargs={"user_timestamp": user_timestamp}
        )
        
        ai_message = ChatMessage(
            content=str(response),
            role=MessageRole.ASSISTANT,
            additional_kwargs={"assistant_timestamp": assistant_timestamp}
        )

        self._memory.put(user_message)
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # Record the Unix timestamp for when the user sends the message
        user_timestamp = int(time.time())

        # get nodes and postprocess them
        nodes = self._get_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(input=message)
        synthesizer = self._get_response_synthesizer(chat_history, streaming=True)

        response = synthesizer.synthesize(message, nodes)
        assert isinstance(response, StreamingResponse)

        # Create the user message with user_timestamp in additional_kwargs
        user_message = ChatMessage(
            content=message,
            role=MessageRole.USER,
            additional_kwargs={"user_timestamp": user_timestamp}
        )

        # Store the user message in memory before streaming starts
        self._memory.put(user_message)

        def wrapped_gen(response: StreamingResponse) -> ChatResponseGen:
            full_response = ""
            assistant_timestamp = None
            for token in response.response_gen:
                full_response += token

                # Record the Unix timestamp when the first token is returned
                if assistant_timestamp is None:
                    assistant_timestamp = int(time.time())

                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            # Create the assistant message with assistant_timestamp in additional_kwargs
            ai_message = ChatMessage(
                content=full_response,
                role=MessageRole.ASSISTANT,
                additional_kwargs={"assistant_timestamp": assistant_timestamp}
            )

            # Store the assistant message in memory
            self._memory.put(ai_message)

        return StreamingAgentChatResponse(
            chat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
            is_writing_to_memory=False,
        )

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # get nodes and postprocess them
        nodes = await self._aget_nodes(message)
        user_timestamp = int(time.time())
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(
            input=message,
        )
        synthesizer = self._get_response_synthesizer(chat_history)

        response = await synthesizer.asynthesize(message, nodes)
        assistant_timestamp = int(time.time())
        
        user_message = ChatMessage(
            content=message,
            role=MessageRole.USER,
            additional_kwargs={"user_timestamp": user_timestamp}
        )
        
        ai_message = ChatMessage(
            content=str(response),
            role=MessageRole.ASSISTANT,
            additional_kwargs={"assistant_timestamp": assistant_timestamp}
        )
        # await self._memory.aput(user_message)
        # await self._memory.aput(ai_message)
        # Use sync put if aput is not available or not truly async
        if hasattr(self._memory, 'aput'):
            try:
                await self._memory.aput(user_message)
                await self._memory.aput(ai_message)
            except TypeError:
                # Fallback to sync if aput isn't actually async
                self._memory.put(user_message)
                self._memory.put(ai_message)
        else:
            self._memory.put(user_message)
            self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
            
        user_timestamp = int(time.time())
        # get nodes and postprocess them
        nodes = await self._aget_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(
            input=message,
        )
        synthesizer = self._get_response_synthesizer(chat_history, streaming=True)

        response = await synthesizer.asynthesize(message, nodes)
        assert isinstance(response, AsyncStreamingResponse)

        async def wrapped_gen(response: AsyncStreamingResponse) -> ChatResponseAsyncGen:
            full_response = ""
            assistant_timestamp = None
            async for token in response.async_response_gen():
                full_response += token

                if assistant_timestamp is None:
                    assistant_timestamp = int(time.time())
                
                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            user_message = ChatMessage(content=message, role=MessageRole.USER)
            ai_message = ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
            await self._memory.aput(user_message)
            await self._memory.aput(ai_message)

        return StreamingAgentChatResponse(
            achat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
            is_writing_to_memory=False,
        )

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history with Unix timestamps for user and assistant."""
        
        history = self._memory.get_all()    

        # Ensure that each message has the correct additional_kwargs with Unix timestamp
        updated_history = []
        for message in history:
            if message.role == MessageRole.USER:
                # Check if user_timestamp exists, if not add it
                if "user_timestamp" not in message.additional_kwargs:
                    message.additional_kwargs["user_timestamp"] = int(time.time())
            elif message.role == MessageRole.ASSISTANT:
                # Check if assistant_timestamp exists, if not add it
                if "assistant_timestamp" not in message.additional_kwargs:
                    message.additional_kwargs["assistant_timestamp"] = int(time.time())

            updated_history.append(message)

        return updated_history