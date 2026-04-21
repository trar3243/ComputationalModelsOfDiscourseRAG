"""
Compute exact match and BERT CLS similarity metrics for a SQuAD evaluation run.

Reads:  squad_testing/output/evaluated_squad_<strategy>_<retrieval>.jsonl
Writes:
  squad_testing/metrics/per_question_<strategy>_<retrieval>.json  — per-question scores
  squad_testing/metrics/aggregate_<strategy>_<retrieval>.json     — aggregate summary

Exact match uses standard SQuAD normalization (lowercase, strip punctuation/articles).
BERT score is cosine similarity of CLS token embeddings from bert-base-uncased,
taking the max over all gold answers. Null generated answers score 0 for both metrics.

Usage (from repo root):
    python squad_testing/scripts/compute_squad_metrics.py [--strategy <strategy>] [--retrieval {filtered,global}]

Available strategies:
    baseline_258_tok  (default)
    most_recent_low_acc_258t_w128_wtd
    most_recent_258t_w254
    nonlinear_258t_w254
"""

import sys
import os
import json
import re
import string
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import torch
from transformers import BertTokenizerFast, BertModel

DEFAULT_STRATEGY  = "baseline_258_tok"
BERT_MODEL_NAME   = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(generated: str | None, gold_answers: list[str]) -> int:
    if not generated:
        return 0
    norm_gen = _normalize(generated)
    return int(any(_normalize(g) == norm_gen for g in gold_answers))


# ---------------------------------------------------------------------------
# BERT CLS score
# ---------------------------------------------------------------------------

def _cls_embedding(text: str, tokenizer, model, device) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token


def bert_cls_score(generated: str | None, gold_answers: list[str], tokenizer, model, device) -> float:
    if not generated:
        return 0.0
    gen_emb = _cls_embedding(generated, tokenizer, model, device)
    scores = []
    for gold in gold_answers:
        gold_emb = _cls_embedding(gold, tokenizer, model, device)
        cos = torch.nn.functional.cosine_similarity(gen_emb, gold_emb).item()
        scores.append(cos)
    return max(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute SQuAD RAG evaluation metrics.")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Chunking strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--retrieval",
        choices=["filtered", "global"],
        default="filtered",
        help="Retrieval mode used during evaluation (default: filtered)",
    )
    args = parser.parse_args()

    input_path     = f"squad_testing/output/evaluated_squad_{args.strategy}_{args.retrieval}.jsonl"
    per_q_path     = f"squad_testing/metrics/per_question_{args.strategy}_{args.retrieval}.json"
    aggregate_path = f"squad_testing/metrics/aggregate_{args.strategy}_{args.retrieval}.json"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Strategy:  {args.strategy}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Input:     {input_path}")

    print(f"\nLoading {BERT_MODEL_NAME}...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    model     = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    print("Computing metrics...\n")

    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    per_question = []
    for i, ex in enumerate(examples):
        generated = ex.get("generated_answer")
        gold      = ex.get("gold_answers", [])

        em    = exact_match(generated, gold)
        bert  = bert_cls_score(generated, gold, tokenizer, model, device)

        per_question.append({
            "id":               ex.get("id"),
            "title":            ex.get("title"),
            "question":         ex.get("question"),
            "gold_answers":     gold,
            "generated_answer": generated,
            "exact_match":      em,
            "bert_cls_score":   round(bert, 6),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(examples)}]")

    n        = len(per_question)
    n_null   = sum(1 for e in per_question if e["generated_answer"] is None)
    mean_em  = sum(e["exact_match"] for e in per_question) / n if n else 0.0
    mean_bert = sum(e["bert_cls_score"] for e in per_question) / n if n else 0.0

    aggregate = {
        "strategy":         args.strategy,
        "retrieval":        args.retrieval,
        "n_questions":      n,
        "n_null_answers":   n_null,
        "mean_exact_match": round(mean_em, 6),
        "mean_bert_cls":    round(mean_bert, 6),
    }

    with open(per_q_path, "w", encoding="utf-8") as f:
        json.dump(per_question, f, indent=2, ensure_ascii=False)

    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  mean exact match : {mean_em:.4f}")
    print(f"  mean BERT CLS    : {mean_bert:.4f}")
    print(f"  null answers     : {n_null}/{n}")
    print(f"\nWrote: {per_q_path}")
    print(f"Wrote: {aggregate_path}")


if __name__ == "__main__":
    main()
