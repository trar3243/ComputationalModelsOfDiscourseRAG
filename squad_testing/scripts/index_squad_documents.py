"""
Index SQuAD baseline chunks into a dedicated ChromaDB collection.

Reads pre-chunked JSONs from: squad_testing/chunks/baseline_258_tok/
Writes to ChromaDB at:        ./squad_chroma_db/  (collection: "squad_documents")

Kept isolated from the LOONG collection so the two corpora don't interfere.

Usage (from repo root):
    python squad_testing/scripts/index_squad_documents.py
"""

import sys
import os
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from index_documents import LocalEmbeddingFunction, index_documents, return_prechunked_texts

SQUAD_CHROMA_DB_PATH = "./squad_chroma_db"
SQUAD_COLLECTION_NAME = "squad_documents"
SQUAD_CHUNKS_DIR = "squad_testing/chunks/baseline_258_tok"


def create_squad_collection(embedding_model: SentenceTransformer) -> chromadb.Collection:
    embed_fn = LocalEmbeddingFunction(embedding_model)
    client = chromadb.PersistentClient(path=SQUAD_CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=SQUAD_COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return collection


def load_chunks_recursive(base_dir: str):
    """Load all chunks_output.json files found recursively under base_dir."""
    all_chunks, all_ids, all_metadatas = [], [], []

    json_files = glob.glob(os.path.join(base_dir, "**", "chunks_output.json"), recursive=True)
    if not json_files:
        print(f"No chunks_output.json files found under '{base_dir}'.")
        return all_chunks, all_ids, all_metadatas

    import json
    for filepath in sorted(json_files):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  [SKIP] Could not parse {filepath}")
                continue

        source_file = (
            data.get("source_file_name")
            or data.get("metadata", {}).get("source_file", os.path.basename(filepath))
        )
        chunks = data.get("chunks", [])

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_file}_chunk_{i}")
            all_metadatas.append({"source": source_file, "chunk_index": i})

    print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} files.")
    return all_chunks, all_ids, all_metadatas


def main():
    print("=" * 60)
    print("Index SQuAD Chunks into ChromaDB")
    print("=" * 60)

    print(f"\n[1/3] Loading pre-chunked JSONs from '{SQUAD_CHUNKS_DIR}'...")
    chunks, ids, metadatas = load_chunks_recursive(SQUAD_CHUNKS_DIR)
    if not chunks:
        print("No chunks found. Run generate_squad_baseline_chunks.py first.")
        return

    print(f"\n[2/3] Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = create_squad_collection(embedding_model)

    print(f"\n[3/3] Indexing into ChromaDB collection '{SQUAD_COLLECTION_NAME}'...")
    index_documents(collection, chunks, ids, metadatas)

    print(f"\nDone! {collection.count()} chunks indexed. Run evaluate_squad.py to evaluate.")


if __name__ == "__main__":
    main()
