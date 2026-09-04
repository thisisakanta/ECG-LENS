"""
Compute lexical metrics from tokenized hypotheses/references pickles.

Works for any dataset (PTB-XL in-domain or MIMIC-IV cross-domain) as long as the
two pickles are in the standard format dumped by the generation pipeline:
    <hyp>.pkl : List[List[str]]
    <ref>.pkl : List[List[List[str]]]

Usage (from the repository root):
    python scripts/run_lexical.py \
        --hyp mimic_out/transformer_tokenized_hypotheses.pkl \
        --ref mimic_out/transformer_tokenized_references.pkl \
        --tag MIMIC-IV --out mimic_out/lexical_metrics.csv
"""

import argparse
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from lexical import compute_lexical_metrics


def main():
    ap = argparse.ArgumentParser(description="Lexical metrics from tokenized pickles")
    ap.add_argument("--hyp", required=True, help="tokenized hypotheses pickle")
    ap.add_argument("--ref", required=True, help="tokenized references pickle")
    ap.add_argument("--tag", default="dataset", help="label for this run (e.g. MIMIC-IV)")
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    with open(args.hyp, "rb") as f:
        hyp = pickle.load(f)
    with open(args.ref, "rb") as f:
        ref = pickle.load(f)

    metrics = compute_lexical_metrics(hyp, ref)

    print(f"\n===== Lexical metrics [{args.tag}] (backend={metrics['backend']}, "
          f"N={metrics['n_samples']}) =====")
    order = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1-P", "ROUGE-1-R",
             "ROUGE-1-F", "ROUGE-L", "METEOR"]
    for k in order:
        print(f"  {k:12s} {metrics[k]:.4f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset"] + order)
            w.writerow([args.tag] + [f"{metrics[k]:.4f}" for k in order])
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
