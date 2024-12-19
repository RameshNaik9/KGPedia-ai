import os
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.gemini import Gemini
from pinecone import Pinecone
from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import QueryFusionRetriever
from query_fusion_retriever import QueryFusionRetriever
import Stemmer

from utils import measure_time

load_dotenv()

class KGPediaModel:
    """
    Responsible for configuring and providing models and embeddings.
    Adheres to SRP by focusing solely on model management.
    """
    _pinecone_client = None

    def __init__(self, pinecone_api_key=None, google_api_key=None):
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.pinecone_api_key or not self.google_api_key:
            raise ValueError("No API keys??")

        # Initialize Pinecone client only if not already initialized
        if KGPediaModel._pinecone_client is None:
            KGPediaModel._pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.pc = KGPediaModel._pinecone_client
        
        self.base_persist_dir = "pinecone index"
        
        # Map from chat_profile to host_url
        self.chat_profile_url_map = {
            "Career": os.getenv("CAREER"),
            "Academic": os.getenv("ACADEMIC"),
            "Bhaat": os.getenv("BHAAT"),
            "Gymkhana": os.getenv("GYMKHANA"),
        }
        self.configure_models()

    def configure_models(self):
        """Configure the embedding and LLM models."""
        Settings.embed_model = GeminiEmbedding(
            model_name=os.getenv("embedding_model_name"),
            api_key=self.google_api_key
        )
        Settings.llm = Gemini(
            model_name=os.getenv("llm_model_name"),
            temperature=1,
            api_key=self.google_api_key
        )

    def get_model(self):
        """Return the configured LLM model."""
        return Settings.llm

    def get_embedding_model(self):
        """Return the configured embedding model."""
        return Settings.embed_model

    def get_pinecone_index(self, chat_profile):
        """Retrieve the Pinecone index based on the chat_profile."""
        host_url = self.chat_profile_url_map.get(chat_profile)
        if not host_url:
            raise ValueError(f"Host URL for chat profile {chat_profile} not found!")
        return self.pc.Index(host=host_url)

    def get_vector_store(self, chat_profile):
        """Get the vector store based on the chat_profile."""
        pinecone_index = self.get_pinecone_index(chat_profile)
        return PineconeVectorStore(pinecone_index=pinecone_index)

    def load_vector_index(self, chat_profile):
        """Load the vector index for a specific chat profile."""
        persist_dir = os.path.join(self.base_persist_dir, chat_profile)
        vector_store = self.get_vector_store(chat_profile)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=vector_store
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index
    
    def get_storage_context(self, chat_profile):
        """Get the storage context for a specific chat profile."""
        persist_dir = os.path.join(self.base_persist_dir, chat_profile)
        vector_store = self.get_vector_store(chat_profile)
        return StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=vector_store,
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir)
        )

    @measure_time("Fusion retriever creation time")
    def get_fusion_retriever(self, chat_profile, similarity_top_k=4):
        """Get a fusion retriever that combines vector and BM25 retrieval."""
        # persist_dir = os.path.join(self.base_persist_dir, chat_profile)
        index = self.load_vector_index(chat_profile)
        # vector_store = self.get_vector_store(chat_profile)
        storage_context = self.get_storage_context(chat_profile)
        
        # Create retrievers
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=storage_context.docstore,
            similarity_top_k=similarity_top_k,
            stemmer=Stemmer.Stemmer("english"),
            verbose=True
        )
        
        # Create fusion retriever
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True
        )
        
        return fusion_retriever