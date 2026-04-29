"""
Evaluate the SQuAD RAG pipeline on the SQuAD validation set.

For each SQuAD validation question:
  1. Retrieve the top-k chunks from ChromaDB.
  2. Generate an answer with Gemini.
  3. Append result to squad_testing/output/evaluated_squad_<strategy>_<retrieval>.jsonl (idempotent).

Usage (from repo root):
    python squad_testing/scripts/evaluate_squad.py [--strategy <strategy>] [--retrieval {filtered,global}]

Available strategies:
    baseline_258_tok  (default)
    most_recent_low_acc_258t_w128_wtd
    most_recent_258t_w254
    nonlinear_258t_w254

Retrieval modes:
    filtered  (default) — retrieve only from the gold article's chunks
    global              — retrieve from all chunks in the collection

Optional: set N_RESULTS and LIMIT to control retrieval depth and how many questions to run.
"""

import sys
import os
import json
import re
import time
import argparse
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from datasets import load_dataset
import anthropic
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, ANTHROPIC_API_KEY, GENERATION_MODEL
from index_documents import LocalEmbeddingFunction

DEFAULT_STRATEGY = "baseline_258_tok"
N_RESULTS        = 10   # chunks to retrieve per question
DEFAULT_LIMIT    = 1000
SAMPLE_SEED      = 42


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def load_collection(strategy: str) -> chromadb.Collection:
    chroma_path = f"./squad_chroma_db_{strategy}"
    collection_name = f"squad_{strategy}"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = LocalEmbeddingFunction(model)
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=collection_name, embedding_function=embed_fn)


def retrieve(collection, query: str, source_file: str, n_results: int, filtered: bool) -> tuple[list[str], list[int]]:
    kwargs = dict(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    if filtered:
        kwargs["where"] = {"source": {"$in": [source_file]}}
    results = collection.query(**kwargs)
    if not results["documents"]:
        return [], []
    docs = results["documents"][0]
    indices = [m.get("chunk_index") for m in results["metadatas"][0]]
    return docs, indices


def generate_answer(client: anthropic.Anthropic, context: str, question: str) -> str:
    while True:
        try:
            response = client.messages.create(
                model=GENERATION_MODEL,
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
                            "Reply with a short phrase or a few words — no explanations, no citations, no 'According to...'.\n"
                            "If the answer is not in the context, reply with exactly: null\n\n"
                            f"CONTEXT:\n{context}\n\n"
                            f"QUESTION: {question}\n\n"
                            "ANSWER:"
                        ),
                    }
                ],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            print("  [Rate limit] sleeping 30s...")
            time.sleep(30)
        except anthropic.APIStatusError as e:
            if e.status_code in (529, 503):
                print("  [Server busy] sleeping 30s...")
                time.sleep(30)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQuAD RAG pipeline.")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Chunking strategy to evaluate (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--retrieval",
        choices=["filtered", "global"],
        default="filtered",
        help="filtered: retrieve only from the gold article's chunks; global: retrieve from all chunks (default: filtered)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of questions to evaluate (default: {DEFAULT_LIMIT}). Pass 0 for full dataset.",
    )
    args = parser.parse_args()
    limit = args.limit if args.limit > 0 else None

    filtered = args.retrieval == "filtered"
    output_path = f"squad_testing/output/evaluated_squad_{args.strategy}_{args.retrieval}.jsonl"

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to .env and run `source ./.env` from the repo root first.")

    print(f"Strategy:  {args.strategy}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Limit:     {limit if limit else 'full dataset'}")
    print(f"Output:    {output_path}")

    print("Loading SQuAD validation set...")
    squad = list(load_dataset("rajpurkar/squad", split="train"))
    random.seed(SAMPLE_SEED)
    random.shuffle(squad)
    if limit:
        squad = squad[:limit]

    print("Loading ChromaDB collection...")
    collection = load_collection(args.strategy)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Resume from previous run
    processed_ids: set[str] = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "id" in obj:
                            processed_ids.add(obj["id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Resuming: {len(processed_ids)} already done.\n")

    total = len(squad)
    counter = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for example in squad:
            qid = example["id"]
            if qid in processed_ids:
                counter += 1
                continue

            title       = example["title"]
            question    = example["question"]
            gold        = example["answers"]["text"]   # list of acceptable answers
            source_file = f"{slugify(title.replace('_', ' '))}.txt"

            chunks, chunk_indices = retrieve(collection, question, source_file, N_RESULTS, filtered)
            if not chunks:
                # Article not in corpus — skip gracefully
                result = {
                    "id": qid,
                    "title": title,
                    "question": question,
                    "gold_answers": gold,
                    "generated_answer": None,
                    "note": "article not found in corpus",
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                processed_ids.add(qid)
                counter += 1
                continue

            context = "\n\n---\n\n".join(chunks)
            generated = generate_answer(client, context, question)

            result = {
                "id": qid,
                "title": title,
                "question": question,
                "gold_answers": gold,
                "generated_answer": generated,
                "retrieved_chunk_indices": chunk_indices,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            processed_ids.add(qid)

            counter += 1
            print(f"  [{counter}/{total}] {title!r}: {question[:60]}")

    print(f"\nDone. Results written to '{output_path}'.")


if __name__ == "__main__":
    main()
