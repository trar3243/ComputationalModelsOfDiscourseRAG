"""
Shared configuration for the RAG system, including API keys, model names, and paths for documents and databases.
Edit all parameters here to configure the system. Make sure to set the GEMINI_API_KEY in your .env file for generation to work.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API (used only for generation, not embeddings)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Change this based on your quota
GENERATION_MODEL = "gemini-2.5-flash"
#GENERATION_MODEL = "gemma-4-31b-it"
#GENERATION_MODEL = "gemma-3-27b-it"
#GENERATION_MODEL = "gemini-3.1-flash-lite-preview"

# Local embedding model (no API key needed and same as the one used usually for embedding generation in the ChromaDB cloud version)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# ChromaDB persistent configuration -- If we want to use the cloud, we need to change the client and provide the appropriate API key and URL
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"

# Documents directory where we store .txt documents
# DOCUMENTS_DIR = "./documents"
DOCUMENTS_DIR = "./documents/loong_docs"
# DOCUMENTS_DIR= "./chunks/'method=most_recent_target_size_tokens=258_search_window=254_full_sent=False_weighted=False'"
# Chunking hyperparameters for splitting docs
# not neeeded for our corefernce tests
CHUNK_SIZE = 258
CHUNK_OVERLAP = 20
