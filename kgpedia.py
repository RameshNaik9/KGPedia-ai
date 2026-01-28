import os
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.embeddings.gemini import GeminiEmbedding
from google.genai import types
from llama_index.embeddings.google_genai.base import GoogleGenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import MarkdownNodeParser
# from llama_index.llms.gemini import Gemini
from llama_index.llms.google_genai import GoogleGenAI
from pinecone import Pinecone
from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
# from llama_index.core.retrievers import QueryFusionRetriever
from query_fusion_retriever import QueryFusionRetriever, FUSION_MODES
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
        embedding_model_name = os.getenv("embedding_model_name")
        llm_model_name = os.getenv("llm_model_name")
        
        if not embedding_model_name or not llm_model_name:
            raise ValueError("No model config??")
        
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name=embedding_model_name,
            api_key=self.google_api_key, #type:ignore
            embedding_config=types.EmbedContentConfig(output_dimensionality=768)
        )
        Settings.llm = GoogleGenAI(
            model=llm_model_name,
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
    
    def upsert_new_data(self, chat_profile: str, directory_path: str, clear_existing: bool = False):
        """
        Upsert new data into the Pinecone index for a specific chat profile.
        
        Args:
            chat_profile: One of 'Career', 'Academic', 'Bhaat', 'Gymkhana'
            directory_path: Path to folder containing Markdown (.md) files
            clear_existing: If True, deletes all existing vectors before upserting (full retrain)
        
        Returns:
            dict: Summary of the upsert operation
        """
        persist_dir = os.path.join(self.base_persist_dir, chat_profile)
        
        # Validate directory
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        # Get Pinecone index and vector store
        pinecone_index = self.get_pinecone_index(chat_profile)
        vector_store = self.get_vector_store(chat_profile)
        
        # Check dimension compatibility
        pinecone_dim = pinecone_index.describe_index_stats().dimension
        embed_model = Settings.embed_model
        # Generate a test embedding to check dimension
        test_embedding = embed_model.get_text_embedding("test")
        embed_dim = len(test_embedding)
        
        if embed_dim != pinecone_dim:
            print(f"\n⚠️  DIMENSION MISMATCH WARNING!")
            print(f"   Embedding model output dimension: {embed_dim}")
            print(f"   Pinecone index dimension: {pinecone_dim}")
            print(f"   These must match for upsertion to succeed.")
            print(f"   Either:")
            print(f"     1. Set output_dimensionality={pinecone_dim} in GoogleGenAIEmbedding's base.py as config = types.EmbedContentConfig(output_dimensionality={pinecone_dim})")
            print(f"     2. Recreate Pinecone index with dimension={embed_dim}")
            raise ValueError(f"Vector dimension {embed_dim} does not match Pinecone index dimension {pinecone_dim}")
        
        print(f"✅ Dimension check passed: {embed_dim} (embedding) == {pinecone_dim} (Pinecone)")
        
        # Optionally clear existing data (for full retrain)
        if clear_existing:
            print(f"🗑️  Clearing existing vectors from '{chat_profile}' index...")
            pinecone_index.delete(delete_all=True)
        
        # Load Markdown files
        print(f"📂 Loading Markdown files from '{directory_path}'...")
        documents = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".md"],
            recursive=True,
        ).load_data()
        
        if not documents:
            raise ValueError(f"No Markdown files found in '{directory_path}'.")
        
        print(f"📄 Loaded {len(documents)} Markdown document(s).")
        
        # Parse documents using MarkdownNodeParser
        print("🔧 Parsing documents with MarkdownNodeParser...")
        markdown_parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )
        nodes = markdown_parser.get_nodes_from_documents(documents)
        print(f"📦 Created {len(nodes)} structured nodes from Markdown headers.")
        
        # Create storage context and upsert
        print("🚀 Upserting nodes into Pinecone (this may take a moment)...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
        # Persist docstore locally (required for BM25 retriever)
        print(f"💾 Saving local docstore to '{persist_dir}'...")
        os.makedirs(persist_dir, exist_ok=True)
        storage_context.persist(persist_dir=persist_dir)
        
        summary = {
            "chat_profile": chat_profile,
            "documents_loaded": len(documents),
            "nodes_upserted": len(nodes),
            "persist_dir": persist_dir,
            "clear_existing": clear_existing,
        }
        
        print(f"✅ Upsert complete for '{chat_profile}'!")
        print(f"   Documents: {len(documents)}")
        print(f"   Nodes: {len(nodes)}")
        print(f"   Local persist: {persist_dir}")
        
        return summary
    
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
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            verbose=True
        )
        
        return fusion_retriever