"""
Generate f-coref coreference clusters for all 412 SQuAD Wikipedia articles.

Reads:  squad_testing/data/squad_wiki_full_articles.parquet (columns: title, text)
Writes: clusters/squad/<slugified_title>.json

Output format: List[List[List[int, int]]] — character-span mentions, same
format as clusters/financial/*.json.

Usage (from repo root):
    python squad_testing/scripts/generate_squad_clusters.py [--limit N]
"""
import json
import re
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from ClassDefinition.segmenter import Segmenter

PARQUET_PATH = REPO_ROOT / "squad_testing" / "data" / "squad_wiki_full_articles.parquet"
OUTPUT_DIR   = REPO_ROOT / "clusters" / "squad"


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate f-coref clusters for SQuAD articles.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max articles to process (0 = all).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    articles = list(df.itertuples(index=False))
    if args.limit > 0:
        articles = articles[:args.limit]

    print(f"Articles to process: {len(articles)}")
    print(f"Output dir: {OUTPUT_DIR}")

    import torch
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    segmenter = Segmenter(device=device, tokenizer=tokenizer)

    skipped = 0
    processed = 0

    for row in articles:
        title = row.title
        text  = row.text
        slug  = slugify(title)
        out_path = OUTPUT_DIR / f"{slug}.json"

        if out_path.exists():
            skipped += 1
            continue

        print(f"\n[{processed + skipped + 1}/{len(articles)}] {title!r}")
        clusters = segmenter.get_clusters(text)

        serializable = [
            [[int(start), int(end)] for start, end in cluster]
            for cluster in clusters
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

        processed += 1
        print(f"  -> {len(clusters)} clusters written to {out_path.name}")

    print(f"\nDone. Processed: {processed}, Skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()
