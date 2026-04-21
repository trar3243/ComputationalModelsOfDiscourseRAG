"""
Index SQuAD chunks into a dedicated ChromaDB collection.

Reads pre-chunked JSONs from: squad_testing/chunks/<strategy>/
Writes to ChromaDB at:        ./squad_chroma_db_<strategy>/  (collection: "squad_<strategy>")

Kept isolated from the LOONG collection so the two corpora don't interfere.

Usage (from repo root):
    python squad_testing/scripts/index_squad_documents.py [--chunks-dir squad_testing/chunks/<strategy>]

Available strategies (squad_testing/chunks/):
    baseline_258_tok  (default)
    most_recent_low_acc_258t_w128_wtd
    most_recent_258t_w254
    nonlinear_258t_w254
"""

import sys
import os
import glob
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from index_documents import LocalEmbeddingFunction, index_documents, return_prechunked_texts

DEFAULT_CHUNKS_DIR = "squad_testing/chunks/baseline_258_tok"


def collection_name_from_dir(chunks_dir: str) -> str:
    strategy = Path(chunks_dir).name
    return f"squad_{strategy}"


def chroma_path_from_dir(chunks_dir: str) -> str:
    strategy = Path(chunks_dir).name
    return f"./squad_chroma_db_{strategy}"


def create_squad_collection(embedding_model: SentenceTransformer, chroma_path: str, collection_name: str) -> chromadb.Collection:
    embed_fn = LocalEmbeddingFunction(embedding_model)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )
    return collection


def load_chunks_recursive(base_dir: str):
    """Load all *.json files found recursively under base_dir."""
    all_chunks, all_ids, all_metadatas = [], [], []

    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    if not json_files:
        print(f"No .json files found under '{base_dir}'.")
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
        # Normalize to "Article Name.txt" format to match evaluate_squad.py
        source_file = source_file.replace("_", " ")
        if not source_file.endswith(".txt"):
            source_file += ".txt"
        chunks = data.get("chunks", [])

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_file}_chunk_{i}")
            all_metadatas.append({"source": source_file, "chunk_index": i})

    print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} files.")
    return all_chunks, all_ids, all_metadatas


def main():
    parser = argparse.ArgumentParser(description="Index SQuAD chunks into ChromaDB.")
    parser.add_argument(
        "--chunks-dir",
        default=DEFAULT_CHUNKS_DIR,
        help=f"Path to chunked JSONs directory (default: {DEFAULT_CHUNKS_DIR})",
    )
    args = parser.parse_args()

    chunks_dir = args.chunks_dir
    collection_name = collection_name_from_dir(chunks_dir)
    chroma_path = chroma_path_from_dir(chunks_dir)

    print("=" * 60)
    print("Index SQuAD Chunks into ChromaDB")
    print(f"  strategy:   {Path(chunks_dir).name}")
    print(f"  collection: {collection_name}")
    print(f"  chroma db:  {chroma_path}")
    print("=" * 60)

    print(f"\n[1/3] Loading pre-chunked JSONs from '{chunks_dir}'...")
    chunks, ids, metadatas = load_chunks_recursive(chunks_dir)
    if not chunks:
        print("No chunks found.")
        return

    print(f"\n[2/3] Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = create_squad_collection(embedding_model, chroma_path, collection_name)

    print(f"\n[3/3] Indexing into ChromaDB collection '{collection_name}'...")
    index_documents(collection, chunks, ids, metadatas)

    print(f"\nDone! {collection.count()} chunks indexed. Run evaluate_squad.py --strategy {Path(chunks_dir).name} to evaluate.")


if __name__ == "__main__":
    main()
