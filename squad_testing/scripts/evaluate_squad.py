"""
Evaluate the SQuAD RAG pipeline on the SQuAD validation set.

For each SQuAD validation question:
  1. Retrieve the top-k chunks from ChromaDB, filtered to the article for that question.
  2. Generate an answer with Gemini.
  3. Append result to squad_testing/evaluated_squad.jsonl (idempotent).

Usage (from repo root):
    python squad_testing/scripts/evaluate_squad.py

Optional: set N_RESULTS and LIMIT to control retrieval depth and how many questions to run.
"""

import sys
import os
import json
import re
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import chromadb
from datasets import load_dataset
from google.genai import errors
import google.genai as genai
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY, GENERATION_MODEL
from index_documents import LocalEmbeddingFunction

SQUAD_CHROMA_DB_PATH  = "./squad_chroma_db"
SQUAD_COLLECTION_NAME = "squad_documents"
OUTPUT_PATH           = "squad_testing/evaluated_squad.jsonl"
N_RESULTS             = 10    # chunks to retrieve per question
LIMIT                 = 20  # set to an int (e.g. 100) to stop early; None = full validation set


def slugify(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "-", title).strip()


def load_collection() -> chromadb.Collection:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = LocalEmbeddingFunction(model)
    client = chromadb.PersistentClient(path=SQUAD_CHROMA_DB_PATH)
    return client.get_collection(name=SQUAD_COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_filtered(collection, query: str, source_file: str, n_results: int) -> tuple[list[str], list[int]]:
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"source": {"$in": [source_file]}},
        include=["documents", "metadatas"],
    )
    if not results["documents"]:
        return [], []
    docs = results["documents"][0]
    indices = [m.get("chunk_index") for m in results["metadatas"][0]]
    return docs, indices


def generate_answer(client_genai, context: str, question: str) -> str:
    prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question. "
        "Be concise.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    while True:
        try:
            response = client_genai.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
            )
            return response.text
        except (errors.ClientError, errors.ServerError) as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("  [Rate limit] sleeping 35s...")
                time.sleep(35)
            elif "503" in msg or "UNAVAILABLE" in msg:
                print("  [Server busy] sleeping 60s...")
                time.sleep(60)
            else:
                raise


def main():
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set. Run `source ./.env` from the repo root first.")

    print("Loading SQuAD validation set...")
    squad = load_dataset("rajpurkar/squad", split="train")

    print("Loading ChromaDB collection...")
    collection = load_collection()

    client_genai = genai.Client(api_key=GEMINI_API_KEY)

    # Resume from previous run
    processed_ids: set[str] = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "id" in obj:
                            processed_ids.add(obj["id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Resuming: {len(processed_ids)} already done.\n")

    total = min(len(squad), LIMIT) if LIMIT else len(squad)
    counter = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, example in enumerate(squad):
            if LIMIT and i >= LIMIT:
                break

            qid = example["id"]
            if qid in processed_ids:
                counter += 1
                continue

            title       = example["title"]
            question    = example["question"]
            gold        = example["answers"]["text"]   # list of acceptable answers
            source_file = f"{slugify(title.replace('_', ' '))}.txt"

            chunks, chunk_indices = retrieve_filtered(collection, question, source_file, N_RESULTS)
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
            generated = generate_answer(client_genai, context, question)

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

    print(f"\nDone. Results written to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
