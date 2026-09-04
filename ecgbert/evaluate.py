"""
Config-driven ECG-BERT evaluation (evaluation-only).

Switch the diagnostic segment by pointing --config at a different YAML in
configs/ (super / sub / rhythm / form) -- no source edits required. Each segment
has its own class list, its own trained checkpoint, and computes its own F1.

Workflow (per segment):
    ground-truth reports  --\
                             >-- BERT labeler --> label sets --> F1
    generated reports     --/

Usage
-----
Single segment:
    python evaluate.py --config configs/super.yaml \
        --gt ground_truth_reports.csv --gen generated_reports.csv

All four segments at once (each with its own config + checkpoint):
    python evaluate.py --config configs/super.yaml configs/sub.yaml \
        configs/rhythm.yaml configs/form.yaml \
        --gt ground_truth_reports.csv --gen generated_reports.csv

Both --gt and --gen are CSVs with a report-text column (default
"Report Impression"). Reports are labeled by the segment's BERT model and the
predicted label sets are compared with per-segment F1 (macro / micro / per-class).

Note: requires the trained per-segment checkpoints and the pretrained BERT
referenced in each config; this script performs no training.
"""

import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

import utils
from models.bert_labeler import bert_labeler
from datasets.unlabeled_dataset import UnlabeledDataset

PAD_IDX = 0


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["segment", "num_classes", "checkpoint", "conditions"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    if len(cfg["conditions"]) != cfg["num_classes"]:
        raise ValueError(
            f"{path}: num_classes={cfg['num_classes']} but {len(cfg['conditions'])} conditions listed"
        )
    cfg.setdefault("pretrain_path", None)
    cfg.setdefault("report_column", "Report Impression")
    return cfg


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_segment_model(cfg, device):
    kwargs = {"num_classes": cfg["num_classes"]}
    if cfg.get("pretrain_path"):
        kwargs["pretrain_path"] = cfg["pretrain_path"]
    model = bert_labeler(**kwargs)

    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state = OrderedDict((k[len("module."):] if k.startswith("module.") else k, v)
                        for k, v in state.items())
    model.load_state_dict(state)
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------
def _collate(sample_list):
    tensors = [s["imp"] for s in sample_list]
    batched = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=PAD_IDX)
    lengths = [s["len"] for s in sample_list]
    return {"imp": batched, "len": lengths}


@torch.no_grad()
def label_reports(model, csv_path, num_classes, device, report_column="Report Impression",
                  batch_size=18, num_workers=2):
    """Return a (num_reports, num_classes) 0/1 matrix of predicted positive labels.
    Head classes: 0 blank, 1 positive, 2 negative, 3 uncertain.
    We map positive(1)->1 and everything else->0 (uncertain treated as negative),
    matching the project's existing labeled_reports.csv convention."""
    loader = torch.utils.data.DataLoader(
        UnlabeledDataset(csv_path, report_column), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
    )
    per_class = [[] for _ in range(num_classes)]
    for data in loader:
        batch = data["imp"].to(device)
        attn = utils.generate_attention_masks(batch, data["len"], device)
        out = model(batch, attn)                       # list[num_classes] of (B,4)
        for j in range(num_classes):
            per_class[j].append(out[j].argmax(dim=1).cpu())
    preds = torch.stack([torch.cat(c, dim=0) for c in per_class], dim=1).numpy()  # (N, C)
    return (preds == 1).astype(int)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_segment(cfg, gt_csv, gen_csv, device):
    model = load_segment_model(cfg, device)
    n = cfg["num_classes"]
    col = cfg["report_column"]
    y_true = label_reports(model, gt_csv, n, device, report_column=col)
    y_pred = label_reports(model, gen_csv, n, device, report_column=col)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"[{cfg['segment']}] row mismatch: gt={y_true.shape[0]} gen={y_pred.shape[0]}. "
            "GT and generated CSVs must be row-aligned (same reports, same order)."
        )
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "segment": cfg["segment"],
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_class": dict(zip(cfg["conditions"], [float(x) for x in per])),
        "n_reports": int(y_true.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser(description="Config-driven ECG-BERT segment evaluation")
    ap.add_argument("--config", nargs="+", required=True, help="One or more segment YAML configs")
    ap.add_argument("--gt", required=True, help="CSV of ground-truth reports")
    ap.add_argument("--gen", required=True, help="CSV of generated reports (row-aligned with --gt)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        print(f"\n===== Evaluating segment: {cfg['segment']} ({cfg['num_classes']} classes) =====")
        res = evaluate_segment(cfg, args.gt, args.gen, device)
        results.append(res)
        print(f"  macro-F1 = {res['macro_f1']:.4f}   micro-F1 = {res['micro_f1']:.4f}   "
              f"(N={res['n_reports']})")
        for cond, s in res["per_class"].items():
            print(f"    {cond}: {s:.4f}")

    if len(results) > 1:
        print("\n===== F1-ECGbert summary (per segment) =====")
        for r in results:
            print(f"  {r['segment']:8s}  macro={r['macro_f1']:.4f}  micro={r['micro_f1']:.4f}")
        print(f"  {'MEAN':8s}  macro={np.mean([r['macro_f1'] for r in results]):.4f}  "
              f"micro={np.mean([r['micro_f1'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
