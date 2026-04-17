"""
filter_non_english_docs.py

Walks the documents/ directory and removes any file whose content is
detected as non-English. Supports --dry-run to preview without deleting.

Usage:
    python filter_non_english_docs.py            # delete non-English files
    python filter_non_english_docs.py --dry-run  # preview only
"""

import argparse
import json
import os
from pathlib import Path

from langdetect import detect, LangDetectException

DOCS_DIR = Path("documents")
EXTENSIONS = {".txt", ".md", ".json"}
SAMPLE_SIZE = 2000  # characters to sample per file


def extract_sample(path: Path) -> str:
    """Return a text sample from a file for language detection."""
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")[:SAMPLE_SIZE]
        # Collect string values recursively
        chunks = []
        def collect(obj):
            if isinstance(obj, str):
                chunks.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)
        collect(data)
        return " ".join(chunks)[:SAMPLE_SIZE]
    else:
        return path.read_text(encoding="utf-8", errors="ignore")[:SAMPLE_SIZE]


def is_english(path: Path) -> bool:
    sample = extract_sample(path).strip()
    if not sample:
        return True  # empty file — leave it alone
    try:
        return detect(sample) == "en"
    except LangDetectException:
        return True  # can't detect — leave it alone


def main():
    parser = argparse.ArgumentParser(description="Remove non-English files from documents/")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    removed, kept = [], []

    for path in sorted(DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        if is_english(path):
            kept.append(path)
        else:
            removed.append(path)
            if not args.dry_run:
                path.unlink()

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results:")
    if removed:
        print(f"\n  {'Would remove' if args.dry_run else 'Removed'} ({len(removed)}):")
        for p in removed:
            print(f"    - {p}")
    else:
        print("  No non-English files found.")

    print(f"\n  Kept ({len(kept)}) English files.")


if __name__ == "__main__":
    main()
