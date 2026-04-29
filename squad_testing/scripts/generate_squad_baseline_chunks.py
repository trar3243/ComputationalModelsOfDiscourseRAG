"""
Generate 258-token baseline chunks for all 412 SQuAD Wikipedia articles.

Reads: squad_testing/data/squad_wiki_full_articles.parquet (columns: title, text)
Writes: squad_testing/chunks/baseline_258_tok/{title_slug}/chunks_output.json

JSON format:
  {
      "metadata": { "source_file": "...", "method": "fixed_token_window", ... },
      "chunks": ["...", ...]
  }

Chunking treats each article as a single continuous token stream — no separator
logic. The full text is tokenized, sliced into non-overlapping 258-token windows,
then each window is decoded back to a string.

Usage (from repo root):
    python squad_testing/scripts/generate_squad_baseline_chunks.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PARQUET_PATH   = Path("squad_testing/data/squad_wiki_full_articles.parquet")
OUTPUT_BASE    = Path("squad_testing/chunks/baseline_258_tok")
CHUNK_SIZE     = 258   # tokens

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def chunk_text(text: str) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), CHUNK_SIZE):
        window = token_ids[i : i + CHUNK_SIZE]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
    return chunks


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} articles from {PARQUET_PATH}\n")

    for _, row in df.iterrows():
        title      = row["title"]
        text       = row["text"]
        slug       = slugify(title)
        source_file = f"{slug}.txt"

        chunks = chunk_text(text)

        out_dir = OUTPUT_BASE / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks_output.json"

        output = {
            "metadata": {
                "source_file":             source_file,
                "method":                  "fixed_token_window",
                "chunk_size_tokens":       CHUNK_SIZE,
                "chunk_overlap_tokens":    0,
                "total_chunks_generated":  len(chunks),
            },
            "chunks": chunks,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  {title!r:60s} -> {len(chunks):4d} chunks")

    print(f"\nDone. Chunks saved to '{OUTPUT_BASE}/'")


if __name__ == "__main__":
    main()
