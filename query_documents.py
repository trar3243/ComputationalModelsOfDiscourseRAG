"""
Steps 5-6:
5. Retrieving relevant chunks from ChromaDB
6. Generating an answer using Gemini 3.1 Flash-Lite (it allows for 500 queries daily as of rn)

Usage:
    1. First run: python index_documents.py  (to build the database)
    2. Then run:  python query_documents.py   (to ask questions)
"""

import chromadb
import google.genai as genai
from sentence_transformers import SentenceTransformer


from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)

print(GEMINI_API_KEY)

from index_documents import LocalEmbeddingFunction

client_genai = genai.Client(api_key=GEMINI_API_KEY)


def load_collection() -> chromadb.Collection:
    """Load the existing ChromaDB collection."""
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = LocalEmbeddingFunction(embedding_model)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    print(f"Loaded collection '{COLLECTION_NAME}' with {collection.count()} chunks.\n")
    return collection


# Step 5: Retrieving relevant chunks from ChromaDB

def retrieve(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
) -> dict:
    """Retrieve the most relevant chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results


# Step 6: Generating an answer using Gemini 3.1 Flash-Lite

def generate_answer(query: str, retrieved_docs: dict) -> str:
    """Build a RAG prompt from the user query and retrieved ChromaDB chunks."""
    context_parts = []
    for doc, metadata in zip(
        retrieved_docs["documents"][0], retrieved_docs["metadatas"][0]
    ):
        source = metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
            Use ONLY the information from the context below to answer the question.
            If the context doesn't contain enough information to answer, say so clearly.
            Break your answer up into nicely readable paragraphs.
            Cite which source file(s) your answer comes from.

            We kindly ask you to review the financial statements of the companies provided above and answer the following questions based solely on the information you have seen. 
            If the question involves content not found in the financial statements, you may ignore this part and only answer the other parts.
            
            CONTEXT:
            {context}

            QUESTION: {query}

            ANSWER:"""

    response = client_genai.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text


# Query pipeline

# try with top-10 results
def ask(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
) -> str:
    """Retrieve relevant chunks, then generate an answer."""
    retrieved = retrieve(collection, query, n_results=n_results)

    if not retrieved["documents"][0]:
        return "No relevant documents found for your query."

    return generate_answer(query, retrieved)


# MAIN

def main():
    print("=" * 60)
    print("Step 5-6: Query Documents with Gemini 3.1 Flash-Lite")
    print("=" * 60 + "\n")

    collection = load_collection()

    print("Ask questions about your documents.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        print("\nSearching and generating answer...\n")
        answer = ask(collection, query)
        print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
