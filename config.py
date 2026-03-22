"""
Shared configuration for the RAG system, including API keys, model names, and paths for documents and databases.
Edit all parameters here to configure the system. Make sure to set the GEMINI_API_KEY in your .env file for generation to work.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API (used only for generation, not embeddings)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-3.1-flash-lite-preview"

# Local embedding model (no API key needed and same as the one used usually for embedding generation in the ChromaDB cloud version)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB persistent configuration -- If we want to use the cloud, we need to change the client and provide the appropriate API key and URL
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"

# Documents directory where we store .txt documents
DOCUMENTS_DIR = "./documents"

# Chunking hyperparameters for splitting docs
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
