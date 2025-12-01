from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.postprocessor.colbert_rerank import ColbertRerank
from typing import Optional

#Custom functions
from chat_engine import ContextChatEngine
from template import Template
from chat_title import get_chat_name
from kgpedia import KGPediaModel
from utils import measure_time
from tags import get_tags 
from question_recommendations import question_recommendations
from cache import NodeCache

import tracemalloc
import os
import psutil
# import uvicorn
import time
import logging
import nest_asyncio
# import asyncio

# Initialize tracing for memory usage
tracemalloc.start()

# Apply nest_asyncio for nested loops
nest_asyncio.apply()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load template and language model
template = Template.get_template()
LLM = KGPediaModel().get_model()

# Retrieve port from environment variables (default to 8000 if not set)
port = os.environ.get("PORT", "8000")

# Chat sessions storage
chat_sessions = {}
# chat_profiles = ['Academics','Bhaat','Career','Gymkhana']
chat_profiles = ['Career']

# # ColBERT Reranker setup
# colbert_reranker = ColbertRerank(
#     top_n=3,model="colbert-ir/colbertv2.0",
#     tokenizer="colbert-ir/colbertv2.0",
#     keep_retrieval_score=True
# )

# Pydantic models for request and response
class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str
    chat_profile: str


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str
    chat_title: Optional[str] = None
    tags_list: Optional[list] = None
    questions_list: Optional[list] = None
    time_taken: Optional[float] = None
    retrieved_sources: list[dict]
    retrieved_content: list[str]

retrievers = {}

# Initialize retrievers when the script starts
for profile in chat_profiles:
    start_time = time.time()
    print(f"Initializing fusion retriever for {profile}...")
    kgpedia_model = KGPediaModel()
    fusion_retriever = kgpedia_model.get_fusion_retriever(chat_profile=profile)
    retrievers[profile] = fusion_retriever
    elapsed_time = time.time() - start_time
    print(f"Fusion retriever for {profile} initialized in {elapsed_time:.2f} seconds.")
    
# Health check endpoint
@app.get("/")
def health_check():
    return {
        "message": "FastAPI Chat Assistant is running!",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }    

async def get_chat_engine(conversation_id: str, chat_profile: str) -> ContextChatEngine:
    if conversation_id not in chat_sessions:
        fusion_retriever = retrievers.get(chat_profile)
        if fusion_retriever is None:
            raise ValueError(f"No retriever found for chat profile: {chat_profile}")
        # timings={}
        # Measure memory initialization time
        # chatmemory_time = time.time()
        memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
        # timings['ChatMemoryBuffer initialization'] = time.time() - chatmemory_time

        # Measure fusion retriever creation time
        # start_time = time.time()
        # fusion_retriever = KGPediaModel().get_fusion_retriever(chat_profile=chat_profile)
        # timings['Fusion retriever creation'] = time.time() - start_time

        # Measure chat engine creation time
        # chat_time = time.time()
        chat_engine = ContextChatEngine.from_defaults(
            retriever=fusion_retriever,
            memory=memory,
            system_prompt=template,
            # node_postprocessors=[colbert_reranker],
        )
        # timings['ContextChatEngine initialization'] = time.time() - chat_time

        # Measure node cache initialization time
        # start_time = time.time()
        chat_engine._node_cache = NodeCache(
            max_size=100,  # Cache up to 100 queries
            ttl=3600      # Cache entries expire after 1 hour
        )
        # timings['NodeCache initialization'] = time.time() - start_time

        chat_sessions[conversation_id] = {"engine": chat_engine, "title_generated": False}

        # # Print timing breakdown
        # print("\n🕒 Chat Engine Initialization Timing Breakdown:")
        # print("----------------------------------------")
        # for component, duration in timings.items():
        #     print(f"⏱️ {component:40}: {duration:.2f} seconds")
        # print("----------------------------------------")
        print(f"Chat session {conversation_id} initialized with profile {chat_profile}.")

    return chat_sessions[conversation_id]["engine"]

@measure_time("Chat Engine use chesi retrieval")
@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        
        # Use await with get_chat_engine
        chat_engine = await get_chat_engine(request.conversation_id, request.chat_profile)
        
        timings = {}
        # Use the async version of chat
        response_start = time.time()
        response = await chat_engine.achat(request.user_message)
        timings['response_generation'] = time.time() - response_start

        # Generate title only if it hasn't been generated yet
        title_start = time.time()
        title = None
        if not chat_sessions[request.conversation_id]['title_generated']:
            title = get_chat_name(request.user_message, str(response))
            # title = chat_title(user_message, response)
            chat_sessions[request.conversation_id]['title_generated'] = True  # Set to True after generating title
        timings['title_generation'] = time.time() - title_start
        
        history = chat_engine.chat_history
        tags_list = get_tags(history,LLM)[0]
        questions_list = question_recommendations(history,LLM)[0]
        if len(history)%8 == 0 or len(history)<=2: #added to save tokens instead of generating every time and diverting the main response generation
            questions_start = time.time()
            questions_list, _ = question_recommendations(history, LLM)
            timings['questions_generation'] = time.time() - questions_start
        else:
            questions_list = []
            
        # Use the source_nodes from the response (already retrieved during achat)
        retrieved_nodes = response.source_nodes
        sources = []
        information = []
        for node in retrieved_nodes:
            sources.append(node.metadata)
            information.append(node.text)
        # Create response object
        
        timings['total_time'] = sum(timings.values())
        # print("Timing breakdown:", timings)
        
        print("\n🕒 Component-wise Timing Breakdown:")
        print("----------------------------------------")
        for component, duration in timings.items():
            print(f"⏱️ {component:20}: {duration:.2f} seconds")
        print("----------------------------------------")
        response_data = ChatResponse(
            conversation_id=request.conversation_id,
            assistant_response=str(response),  # Remove newline characters
            chat_title=title,  # Return title only if it was generated
            tags_list = tags_list,
            questions_list = questions_list,
            time_taken = timings['total_time'],
            retrieved_sources=sources,
            retrieved_content = information
        )

        return response_data

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Delete specific chat endpoint
@app.delete("/chat_delete/{conversation_id}")
async def delete_chat(conversation_id: str):
    try:
        if conversation_id in chat_sessions:
            del chat_sessions[conversation_id]
            return {"message": f"The conversation {conversation_id} has been deleted successfully ♻️"}
        else:
            raise KeyError
    except KeyError:
        logger.warning(f"Attempt to delete non-existing conversation ID: {conversation_id}")
        raise HTTPException(
            status_code=404, detail=f"Conversation ID {conversation_id} does not exist or has already been deleted 🗑️"
        )
    except Exception as e:
        logger.error(f"Unexpected error in delete_chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete all chats endpoint
@app.delete("/chats/delete_all")
def delete_all_chats():
    try:
        if chat_sessions:
            deleted_ids = list(chat_sessions.keys())
            chat_sessions.clear()
            logger.info(f"Deleted {len(deleted_ids)} conversation(s): {deleted_ids}")
            return {
                "message": "All chat conversations have been deleted successfully! 😈",
                "deleted_count": len(deleted_ids),
                "deleted_conversation_ids": deleted_ids
            }
        else:
            logger.info("No active chat conversations to delete")
            return {
                "message": "No active chat conversations found to delete 😕",
                "deleted_count": 0,
                "deleted_conversation_ids": []
            }
    except Exception as e:
        logger.error(f"Unexpected error in delete_all_chats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_all_convo_ids")
def get_all_convo_ids():
    chat_sessions_list = list(chat_sessions.keys())
    if len(chat_sessions_list) == 0:
        raise HTTPException(status_code=404, detail="No active chat conversations found 😕")
    return {"convo_ids": list(chat_sessions.keys())}

@app.get("/get_conv_history/{conversation_id}")
def get_conv_history(conversation_id: str):
    try:
        if conversation_id in chat_sessions:
            chat_engine = chat_sessions[conversation_id]["engine"]
            return {"conversation_id": conversation_id, "history": chat_engine.chat_history}
        else:
            raise KeyError
    except KeyError:
        logger.warning(f"Attempt to retrieve history for non-existing conversation ID: {conversation_id}")
        raise HTTPException(
            status_code=404, detail=f"Conversation ID {conversation_id} does not exist or has already been deleted 🗑️"
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_conv_history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Run the app
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=False,factory=True)