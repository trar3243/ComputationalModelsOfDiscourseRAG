"""
Generate baseline chunks for all documents in documents/loong_docs/
using LangChain's RecursiveCharacterTextSplitter (same as index_documents.py).

Output: chunks_baseline/{domain}/{source_file}/chunks_output.json
Format matches the coreference chunks JSONs:
  {
      "metadata": { ... },
      "chunks": [ "chunk text", ... ]
  }

Usage:
    python generate_baseline_chunks.py
"""

import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

LOONG_DOCS_DIR    = Path("documents/loong_docs")
OUTPUT_BASE_DIR   = Path("chunks_baseline")
SUPPORTED_EXTS    = {".txt", ".md"}


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_text(text)


def main():
    doc_paths = sorted(
        p for p in LOONG_DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix in SUPPORTED_EXTS
    )

    if not doc_paths:
        print(f"No documents found in {LOONG_DOCS_DIR}")
        return

    print(f"Found {len(doc_paths)} documents. Chunking...\n")

    for doc_path in doc_paths:
        domain      = doc_path.parent.name   # e.g. 'financial' or 'paper'
        source_file = doc_path.name          # e.g. '2019-avni123118form10k.txt'

        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        out_dir = OUTPUT_BASE_DIR / domain / source_file
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks_output.json"

        output = {
            "metadata": {
                "domain":                domain,
                "source_file":           source_file,
                "method":                "recursive_character",
                "chunk_size_chars":      CHUNK_SIZE,
                "chunk_overlap_chars":   CHUNK_OVERLAP,
                "total_chunks_generated": len(chunks),
            },
            "chunks": chunks,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"  [{domain}] {source_file} -> {len(chunks)} chunks")

    print(f"\nDone. Baseline chunks saved to '{OUTPUT_BASE_DIR}/'")


if __name__ == "__main__":
    main()
