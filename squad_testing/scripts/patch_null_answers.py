"""
Postprocessing script that:
  1. Patches per_question_*.json files: normalizes "null" strings -> null type,
     and zeroes bert_cls_score for any null generated_answer.
  2. Recomputes and overwrites aggregate_*.json files from the patched data.
"""

import json
import re
from pathlib import Path

METRICS_DIR = Path("squad_testing/metrics")


def patch_per_question(records: list[dict]) -> tuple[list[dict], int]:
    """Returns patched records and count of answers that were null."""
    patched = []
    for r in records:
        r = dict(r)
        if r.get("generated_answer") == "null":
            r["generated_answer"] = None
        if r["generated_answer"] is None:
            r["bert_cls_score"] = 0.0
        patched.append(r)
    n_null = sum(1 for r in patched if r["generated_answer"] is None)
    return patched, n_null


def recompute_aggregate(strategy: str, retrieval: str, records: list[dict], n_null: int) -> dict:
    n = len(records)
    return {
        "strategy": strategy,
        "retrieval": retrieval,
        "n_questions": n,
        "n_null_answers": n_null,
        "mean_exact_match": round(sum(r["exact_match"] for r in records) / n, 6) if n else 0.0,
        "mean_bert_cls": round(sum(r["bert_cls_score"] for r in records) / n, 6) if n else 0.0,
    }


def main():
    per_question_files = sorted(METRICS_DIR.glob("per_question_*.json"))
    if not per_question_files:
        print(f"No per_question_*.json files found in {METRICS_DIR}")
        return

    for pq_path in per_question_files:
        # Extract strategy and retrieval from filename: per_question_<strategy>_<retrieval>.json
        stem = pq_path.stem  # e.g. per_question_baseline_258_tok_global
        match = re.match(r"per_question_(.+)_(filtered|global)$", stem)
        if not match:
            print(f"Skipping {pq_path.name}: unexpected filename format")
            continue
        strategy, retrieval = match.group(1), match.group(2)

        records = json.loads(pq_path.read_text())
        patched, n_null = patch_per_question(records)

        pq_path.write_text(json.dumps(patched, indent=2))
        print(f"Patched {pq_path.name}: {n_null} null answers")

        agg = recompute_aggregate(strategy, retrieval, patched, n_null)
        agg_path = METRICS_DIR / f"aggregate_{strategy}_{retrieval}.json"
        agg_path.write_text(json.dumps(agg, indent=2))
        print(f"  -> rewrote {agg_path.name}  (mean_bert_cls={agg['mean_bert_cls']}, n_null={n_null})")


if __name__ == "__main__":
    main()
