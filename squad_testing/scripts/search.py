"""
Interactive search over the SQuAD ChromaDB collection.

Shows the top-k retrieved chunks with their source article, chunk index,
and similarity distance — useful for inspecting retrieval behaviour
before or alongside running the full eval pipeline.

Usage (from repo root):
    python squad_testing/scripts/search.py

Commands at the prompt:
    <query>          — search and show top-k chunks
    :k <n>           — change how many results to show (default: 5)
    :filter <title>  — restrict results to one article title
    :nofilter        — clear the article filter
    quit / exit / q  — exit
"""

import sys
import os
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from index_documents import LocalEmbeddingFunction

SQUAD_CHROMA_DB_PATH  = "./squad_chroma_db"
SQUAD_COLLECTION_NAME = "squad_documents"
DEFAULT_K             = 5


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def load_collection() -> chromadb.Collection:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = LocalEmbeddingFunction(model)
    client = chromadb.PersistentClient(path=SQUAD_CHROMA_DB_PATH)
    collection = client.get_collection(name=SQUAD_COLLECTION_NAME, embedding_function=embed_fn)
    print(f"Collection '{SQUAD_COLLECTION_NAME}' loaded — {collection.count()} chunks indexed.\n")
    return collection


def search(collection: chromadb.Collection, query: str, k: int, source_filter: str | None):
    where = {"source": {"$in": [source_filter]}} if source_filter else None
    kwargs = dict(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    divider = "─" * 72

    if not docs:
        print("  (no results)\n")
        return

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source      = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        print(f"\n{divider}")
        print(f"  Rank {rank}  |  {source}  |  chunk #{chunk_index}  |  distance: {dist:.4f}")
        print(divider)
        # Wrap long text at 72 chars for readability
        for line in doc.splitlines():
            while len(line) > 72:
                print("  " + line[:72])
                line = line[72:]
            print("  " + line)
    print(f"\n{divider}\n")


def main():
    collection = load_collection()

    k             = DEFAULT_K
    source_filter = None

    print("SQuAD Chunk Search")
    print("  :k <n>           change result count (current: 5)")
    print("  :filter <title>  restrict to one article")
    print("  :nofilter        clear article filter")
    print("  quit             exit")
    print()

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if line.startswith(":k "):
            try:
                k = int(line.split()[1])
                print(f"  Result count set to {k}.")
            except (IndexError, ValueError):
                print("  Usage: :k <integer>")
            continue

        if line.startswith(":filter "):
            title = line[len(":filter "):].strip()
            source_filter = f"{slugify(title.replace('_', ' '))}.txt"
            print(f"  Filter active: source = '{source_filter}'")
            continue

        if line == ":nofilter":
            source_filter = None
            print("  Filter cleared.")
            continue

        if source_filter:
            print(f"  [Filtering to: {source_filter}]")

        search(collection, line, k, source_filter)


if __name__ == "__main__":
    main()
