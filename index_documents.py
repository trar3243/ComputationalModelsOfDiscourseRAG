"""
4 steps in the RAG pipeline:
1. Loading .txt files from a directory
2. Chunking with LangChain
3. Getting local embeddings for the chunks
4. Storing embeddings in a Crhoma persistent collection

Usage:
- Store all .txt files in the 'documents' directory (create it if it doesn't exist).
- Run:
    python index_documents.py
"""

import os
import glob
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# STEP 1: Loading .txt files from a directory

def load_text_files(directory: str) -> list[dict]:
    documents = []
    # txt_files = glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.md"))
    txt_files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True) + glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    if not txt_files:
        print(f"No .txt files found in '{directory}'.")
        return documents

    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        documents.append({"filename": filename, "content": content})
        print(f"  Loaded: {filename} ({len(content)} chars)")

    print(f"\nTotal files loaded: {len(documents)}")
    return documents


# Step 2: Chunking with LangChain

def chunk_documents(documents: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    # delete this when we have the chunked texts
    # but used for baseline
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['filename']}_chunk_{i}")
            all_metadatas.append({
                "source": doc["filename"],
                "chunk_index": i,
            })

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks, all_ids, all_metadatas


# Step 3: Getting local embeddings for the chunks

class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


def create_chroma_db(
    embedding_model: SentenceTransformer,
) -> chromadb.Collection:
    embed_fn = LocalEmbeddingFunction(embedding_model)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return collection


# Step 4: Storing embeddings in a Crhoma persistent collection

def index_documents(
    collection: chromadb.Collection,
    chunks: list[str],
    ids: list[str],
    metadatas: list[dict],
    batch_size: int = 200,
):
    """
    Skips chunks that are already indexed.
    """
    existing = set(collection.get()["ids"])
    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]

    if not new_indices:
        print("All chunks are already indexed. Skipping.")
        return

    new_chunks = [chunks[i] for i in new_indices]
    new_ids = [ids[i] for i in new_indices]
    new_metadatas = [metadatas[i] for i in new_indices]

    total_batches = (len(new_chunks) + batch_size - 1) // batch_size

    for start in range(0, len(new_chunks), batch_size):
        end = start + batch_size
        batch_num = start // batch_size + 1

        collection.add(
            documents=new_chunks[start:end],
            ids=new_ids[start:end],
            metadatas=new_metadatas[start:end],
        )
        print(f"  Indexed batch {batch_num}/{total_batches} ({min(end, len(new_chunks))}/{len(new_chunks)} chunks)")

    print(f"Indexing complete. Total chunks in collection: {collection.count()}")


# MAIN

def main():
    print("=" * 60)
    print("Step 1-4: Index Documents into ChromaDB")
    print("=" * 60)

    # Step 1: Load documents
    print(f"\n[1/4] Loading .txt files from '{DOCUMENTS_DIR}'...")
    documents = load_text_files(DOCUMENTS_DIR)
    if not documents:
        print(f"\nPlease add .txt files to the '{DOCUMENTS_DIR}' directory and try again.")
        return

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents with LangChain RecursiveCharacterTextSplitter...")
    chunks, ids, metadatas = chunk_documents(documents)

    # Step 3: Set up ChromaDB
    print(f"\n[3/4] Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = create_chroma_db(embedding_model)

    # Step 4: Index chunks
    print("\n[4/4] Indexing chunks into ChromaDB (local embeddings)...")
    index_documents(collection, chunks, ids, metadatas)

    print("\nDone! Documents are indexed. Run 'python query_documents.py' to ask questions.")


if __name__ == "__main__":
    main()
