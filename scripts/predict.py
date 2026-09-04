"""
ECG-LENS inference: trained checkpoint + ECG signal(s) -> generated report(s).

Takes a checkpoint saved by `src/train_transformer_decoder.py` (either
`BEST_checkpoint_*.pth.tar` or `checkpoint_*.pth.tar`), rebuilds the model from the
encoder/decoder configs stored inside it, and decodes a report. No training, no
dataset preparation, and no config file needed beyond the tokenizer.

Input -- exactly one of:
    --record  PATH   a WFDB record without extension (PTB-XL or MIMIC-IV);
                     leads are reordered to I,II,III,aVR,aVL,aVF,V1..V6,
                     resampled to --sampling-rate and cropped/padded to --length
    --npy     PATH   a .npy/.npz array shaped (12, L) for one record or
                     (N, 12, L) for a batch
    --hdf5    DIR    a prepared dataset folder; reads <SPLIT>_DATASET.hdf5
                     (use --split and optionally --index / --limit)

Clinical prompt:
    Checkpoints trained with prompt conditioning need a clinical prompt per record.
    Supply it one of three ways:
      --merl-checkpoint PATH   predict the labels from the signal and build the
                               prompt exactly as in training (needs
                               --scp-statements and the MERL repo, see README)
      --prompt "..."           use one explicit prompt for every record
      --prompts FILE.pkl       a pickled List[str], one prompt per record
    Checkpoints trained without conditioning ignore all of these.

Examples
--------
    # one PTB-XL record, prompt predicted from the signal
    python scripts/predict.py \
        --checkpoint checkpoints/ecglens_ptbxl.pth.tar \
        --encoder data/ptbxl/encoder.pkl \
        --record data/ptbxl/records500/00000/00001_hr \
        --merl-checkpoint checkpoints/merl.pth \
        --scp-statements data/ptbxl/scp_statements.csv

    # the whole prepared PTB-XL test split -> CSV for F1-ECGBERT
    python scripts/predict.py \
        --checkpoint checkpoints/ecglens_ptbxl.pth.tar \
        --encoder data/ptbxl/encoder.pkl \
        --hdf5 data/ptbxl --split TEST \
        --merl-checkpoint checkpoints/merl.pth \
        --scp-statements data/ptbxl/scp_statements.csv \
        --out generated_reports.csv

The CSV is written with a "Report Impression" column so it feeds straight into
`python ecgbert/evaluate.py --gen generated_reports.csv ...`.

Run from the repository root.
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prompting                                             # noqa: E402
from train_transformer_decoder import load_checkpoint_compat  # noqa: E402
# Reuse the checkpoint-to-model reconstruction used by the evaluation pipeline so
# inference and evaluation cannot rebuild the model differently.
from generate_reports import build_model                     # noqa: E402

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------------
# Signal loading
# ----------------------------------------------------------------------
def load_from_record(path, sampling_rate, length):
    """Read one WFDB record into (1, 12, length)."""
    from prepare_mimic import load_wfdb_signal
    return load_wfdb_signal(path, sampling_rate, length)[None, ...]


def load_from_npy(path):
    """Read a .npy/.npz array into (N, 12, L)."""
    arr = np.load(path)
    if hasattr(arr, "files"):  # .npz
        arr = arr[arr.files[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (12, L) or (N, 12, L), got {arr.shape}")
    return arr


def load_from_hdf5(folder, split):
    """Read <split>_DATASET.hdf5 signals into (N, 12, L)."""
    import h5py
    path = os.path.join(folder, f"{split}_DATASET.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h:
        arr = np.asarray(h["secgs"][:], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    return arr


def load_signals(args):
    chosen = [bool(args.record), bool(args.npy), bool(args.hdf5)]
    if sum(chosen) != 1:
        raise SystemExit("Choose exactly one input: --record, --npy, or --hdf5.")

    if args.record:
        signals = load_from_record(args.record, args.sampling_rate, args.length)
        ids = [os.path.basename(args.record)]
    elif args.npy:
        signals = load_from_npy(args.npy)
        ids = [f"{os.path.basename(args.npy)}:{i}" for i in range(len(signals))]
    else:
        signals = load_from_hdf5(args.hdf5, args.split)
        ids = [f"{args.split}:{i}" for i in range(len(signals))]

    if args.index is not None:
        if not 0 <= args.index < len(signals):
            raise SystemExit(f"--index {args.index} out of range (N={len(signals)})")
        signals, ids = signals[args.index:args.index + 1], ids[args.index:args.index + 1]
    if args.limit > 0:
        signals, ids = signals[:args.limit], ids[:args.limit]
    return signals, ids


# ----------------------------------------------------------------------
# Clinical prompts
# ----------------------------------------------------------------------
def prompts_from_merl(signals, args):
    """Predict labels from the signals and render them into training-identical prompts."""
    if not args.scp_statements:
        raise SystemExit("--merl-checkpoint requires --scp-statements "
                         "(path to PTB-XL scp_statements.csv) to know the candidate conditions.")
    from merl_predictor import MerlLabelPredictor

    vocab = prompting.load_label_vocabulary(args.scp_statements)
    predictor = MerlLabelPredictor(
        checkpoint_path=args.merl_checkpoint,
        prompt_json=args.merl_prompt_json or None,
        ecg_model=args.merl_ecg_model,
        threshold=args.merl_threshold,
    )
    active = predictor.predict_active_codes(
        signals, vocab["families"],
        code_to_text=lambda c: prompting.code_to_text(c, vocab),
    )
    maps = vocab["maps"]
    return [
        prompting.format_clinical_prompt(
            row["super"], row["sub"], row["form"], row["rhythm"],
            subclass_description_map=maps["sub"],
            form_description_map=maps["form"],
            rhythm_description_map=maps["rhythm"],
        )
        for row in active
    ]


def resolve_prompts(signals, args, uses_prompt):
    n = len(signals)
    if not uses_prompt:
        if args.merl_checkpoint or args.prompt or args.prompts:
            print("[note] checkpoint was trained without prompt conditioning; "
                  "ignoring the supplied prompt options.")
        return None

    if args.merl_checkpoint:
        prompts = prompts_from_merl(signals, args)
    elif args.prompts:
        with open(args.prompts, "rb") as f:
            prompts = list(pickle.load(f))
    elif args.prompt:
        prompts = [args.prompt] * n
    else:
        raise SystemExit(
            "This checkpoint was trained with clinical-prompt conditioning, so each "
            "record needs a prompt. Pass --merl-checkpoint (with --scp-statements) to "
            "predict one from the signal, or --prompt / --prompts to supply it."
        )

    if len(prompts) != n:
        raise SystemExit(f"got {len(prompts)} prompts for {n} records")
    return prompts


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
@torch.no_grad()
def generate_reports(model, signals, prompts, token_encoder, block_size, args, device):
    """Decode one report per record, mirroring the training-time generation path."""
    eos_idx = token_encoder.vocab["<|endoftext|>"]
    sos_idx = token_encoder.vocab["<|startoftext|>"]
    # validate() seeds generation with <|startoftext|> and allows block_size - 1 new
    # tokens; keep that identical here so decoding matches the reported setting.
    max_new = args.max_new_tokens if args.max_new_tokens else block_size - 1

    reports = []
    for i in range(len(signals)):
        ecg = torch.as_tensor(signals[i], dtype=torch.float32, device=device)
        start = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        # generate() appends to model_input[1] in place, so this must be a list.
        model_input = [ecg, start]
        if prompts is not None:
            model_input.append(prompts[i])

        tokens = model.generate(
            model_input, max_new, eos_idx,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            token_encoder=token_encoder,
        ).squeeze(0)

        text = token_encoder.decode(tokens, skip_special_tokens=True)
        reports.append(text)
    return reports


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate an ECG report from a trained ECG-LENS checkpoint")
    ap.add_argument("--checkpoint", required=True, help="trained .pth / .pth.tar checkpoint")
    ap.add_argument("--encoder", required=True, help="training encoder.pkl (tokenizer)")

    src = ap.add_argument_group("input (choose one)")
    src.add_argument("--record", help="WFDB record path without extension")
    src.add_argument("--npy", help=".npy/.npz array (12, L) or (N, 12, L)")
    src.add_argument("--hdf5", help="folder containing <SPLIT>_DATASET.hdf5")
    src.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    src.add_argument("--index", type=int, default=None, help="single record index")
    src.add_argument("--limit", type=int, default=-1, help="cap the number of records")
    src.add_argument("--sampling-rate", type=int, default=500, help="--record only")
    src.add_argument("--length", type=int, default=5000, help="--record only")

    pr = ap.add_argument_group("clinical prompt")
    pr.add_argument("--merl-checkpoint", help="predict labels from the signal")
    pr.add_argument("--scp-statements", help="PTB-XL scp_statements.csv (with --merl-checkpoint)")
    pr.add_argument("--merl-prompt-json", default=None)
    pr.add_argument("--merl-ecg-model", default="resnet18")
    pr.add_argument("--merl-threshold", type=float, default=0.5)
    pr.add_argument("--prompt", help="one explicit prompt for every record")
    pr.add_argument("--prompts", help="pickled List[str], one prompt per record")

    gen = ap.add_argument_group("decoding")
    gen.add_argument("--top-k", type=int, default=1, help="1 = greedy (matches training)")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--max-new-tokens", type=int, default=0, help="0 = checkpoint block size")
    gen.add_argument("--show-prompt", action="store_true", help="also print the clinical prompt")
    gen.add_argument("--out", default=None, help="write a CSV with a Report Impression column")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.encoder, "rb") as f:
        token_encoder = pickle.load(f)
    token_encoder.vocab_size = len(token_encoder.vocab)

    ckpt = load_checkpoint_compat(args.checkpoint, map_location=device)
    model, block_size, enc_cfg = build_model(ckpt, token_encoder, device)
    uses_prompt = enc_cfg.get("use_scp_label_conditioning", False)

    signals, ids = load_signals(args)
    if signals.shape[1] != enc_cfg["in_chan"]:
        raise SystemExit(f"signal has {signals.shape[1]} channels but the checkpoint "
                         f"expects {enc_cfg['in_chan']}")
    print(f"Loaded {len(signals)} record(s), shape {tuple(signals.shape)}; "
          f"prompt conditioning: {uses_prompt}")

    prompts = resolve_prompts(signals, args, uses_prompt)
    reports = generate_reports(model, signals, prompts, token_encoder,
                               block_size, args, device)

    for i, (rid, report) in enumerate(zip(ids, reports)):
        print(f"\n=== {rid} ===")
        if prompts is not None and args.show_prompt:
            print(f"prompt: {prompts[i]}")
        print(report)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record", "Report Impression"])
            for rid, report in zip(ids, reports):
                w.writerow([rid, report])
        print(f"\nWrote {args.out} ({len(reports)} reports)")


if __name__ == "__main__":
    main()
