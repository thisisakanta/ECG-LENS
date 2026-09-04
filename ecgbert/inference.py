"""
F1-ECGBERT inference: report text -> predicted diagnostic labels.

Runs the trained per-segment labelers over one or more reports and prints the
conditions each report asserts. This is the labeling half of F1-ECGBERT on its
own; use `evaluate.py` when you have reference/generated pairs and want the F1.

Segments come from the same YAML configs as `evaluate.py`, so any subset of
super / sub / rhythm / form can be run, each with its own class list, checkpoint
and `num_classes`. Each checkpoint is loaded once and reused for every report.

Usage (from the repository root)
-------------------------------
    # one report, all four segments
    python ecgbert/inference.py \
        --config configs/ecgbert_eval/super.yaml configs/ecgbert_eval/sub.yaml \
                 configs/ecgbert_eval/rhythm.yaml configs/ecgbert_eval/form.yaml \
        --report "sinus rhythm, inferior myocardial infarction, st depression"

    # a CSV of reports -> one label CSV
    python ecgbert/inference.py --config configs/ecgbert_eval/sub.yaml \
        --csv generated_reports.csv --out predicted_labels.csv

    # reports piped in, one per line
    cat reports.txt | python ecgbert/inference.py --config configs/ecgbert_eval/sub.yaml --stdin
"""

import argparse
import sys

import pandas as pd
import torch

import bert_tokenizer
import utils
from constants import PAD_IDX
from evaluate import load_config, load_segment_model


def read_texts(args):
    """Collect report strings from --report / --csv / --stdin."""
    if args.report:
        return list(args.report), [f"report{i}" for i in range(len(args.report))]
    if args.csv:
        texts = bert_tokenizer.get_impressions_from_csv(args.csv, args.report_column)
        return list(texts), [str(i) for i in range(len(texts))]
    if args.stdin:
        lines = [ln.strip() for ln in sys.stdin if ln.strip()]
        if not lines:
            raise SystemExit("no reports received on stdin")
        return lines, [f"line{i}" for i in range(len(lines))]
    raise SystemExit("Provide reports with --report, --csv, or --stdin.")


def tokenize_texts(texts):
    """Report strings -> BERT id lists, using the same tokenizer as training."""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    series = pd.Series(texts).fillna("").astype(str).str.strip()
    series = series.replace("\n", " ", regex=True).replace(r"\s+", " ", regex=True).str.strip()
    return bert_tokenizer.tokenize(series, tokenizer)


@torch.no_grad()
def predict_segment(model, encoded, num_classes, device, batch_size=18):
    """Return a (num_reports, num_classes) 0/1 matrix; only the positive head counts.

    Mirrors evaluate.label_reports so single-report inference and the reported
    metric agree on how head outputs become labels.
    """
    rows = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        tensors = [torch.LongTensor(ids) for ids in chunk]
        batch = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=PAD_IDX).to(device)
        lengths = [len(ids) for ids in chunk]
        attn = utils.generate_attention_masks(batch, lengths, device)
        out = model(batch, attn)  # list[num_classes] of (B, 4)
        preds = torch.stack([out[j].argmax(dim=1).cpu() for j in range(num_classes)], dim=1)
        rows.append((preds == 1).int())
    return torch.cat(rows, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser(description="Label ECG reports with the F1-ECGBERT labelers")
    ap.add_argument("--config", nargs="+", required=True,
                    help="one or more segment YAML configs (configs/ecgbert_eval/*.yaml)")
    ap.add_argument("--report", nargs="+", help="report text (repeatable)")
    ap.add_argument("--csv", help="CSV of reports")
    ap.add_argument("--report-column", default="Report Impression")
    ap.add_argument("--stdin", action="store_true", help="read reports from stdin, one per line")
    ap.add_argument("--out", help="write predicted labels to this CSV")
    ap.add_argument("--batch-size", type=int, default=18)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    texts, ids = read_texts(args)
    encoded = tokenize_texts(texts)
    print(f"Labeling {len(texts)} report(s) across {len(args.config)} segment(s) on {device}")

    # segment -> {condition: 0/1} per report
    per_report = [{} for _ in texts]
    columns = []
    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        model = load_segment_model(cfg, device)          # loaded once per segment
        preds = predict_segment(model, encoded, cfg["num_classes"], device, args.batch_size)
        for r in range(len(texts)):
            for c, cond in enumerate(cfg["conditions"]):
                per_report[r][(cfg["segment"], cond)] = int(preds[r, c])
        columns += [(cfg["segment"], cond) for cond in cfg["conditions"]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for rid, text, labels in zip(ids, texts, per_report):
        positives = [f"{seg}: {cond}" for (seg, cond), v in labels.items() if v == 1]
        print(f"\n=== {rid} ===")
        print(f"report: {text}")
        if positives:
            for p in positives:
                print(f"  - {p}")
        else:
            print("  (no condition asserted by any segment)")

    if args.out:
        frame = {"Report Impression": texts}
        for seg, cond in columns:
            frame[f"{seg}:{cond}"] = [labels[(seg, cond)] for labels in per_report]
        pd.DataFrame(frame).to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
