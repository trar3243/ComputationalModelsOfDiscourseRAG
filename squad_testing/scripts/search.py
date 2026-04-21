"""
Interactive search over a SQuAD ChromaDB collection.

Shows the top-k retrieved chunks with their source article, chunk index,
and similarity distance — useful for inspecting retrieval behaviour
before or alongside running the full eval pipeline.

Usage (from repo root):
    python squad_testing/scripts/search.py [--strategy <strategy>] [--retrieval {filtered,global}]

Available strategies:
    baseline_258_tok  (default)
    most_recent_low_acc_258t_w128_wtd
    most_recent_258t_w254
    nonlinear_258t_w254

Retrieval modes:
    filtered  (default) — start with a per-article filter active (set via :filter)
    global              — start with no filter (search all chunks)

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
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME
from index_documents import LocalEmbeddingFunction

DEFAULT_STRATEGY = "baseline_258_tok"
DEFAULT_K        = 5


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def load_collection(strategy: str) -> chromadb.Collection:
    chroma_path = f"./squad_chroma_db_{strategy}"
    collection_name = f"squad_{strategy}"
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = LocalEmbeddingFunction(model)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
    print(f"Collection '{collection_name}' loaded — {collection.count()} chunks indexed.\n")
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
    parser = argparse.ArgumentParser(description="Interactive search over a SQuAD ChromaDB collection.")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Chunking strategy to search (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--retrieval",
        choices=["filtered", "global"],
        default="filtered",
        help="filtered: start with per-article filter active; global: start with no filter (default: filtered)",
    )
    args = parser.parse_args()

    collection = load_collection(args.strategy)

    k             = DEFAULT_K
    source_filter = None
    if args.retrieval == "filtered":
        print("  Retrieval mode: filtered (use :filter <title> to set article, :nofilter to clear)")

    if args.retrieval == "global":
        print("  Retrieval mode: global (searching all chunks — use :filter <title> to restrict)")

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
