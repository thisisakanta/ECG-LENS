"""
Prepare MIMIC-IV-ECG (raw WFDB + reports CSV) into the model's dataset format
for CROSS-DOMAIN inference. No training; this only builds the test hdf5 so the
existing trained checkpoint can be run on MIMIC exactly like PTB-XL.

Output: <out_dir>/TEST_DATASET.hdf5 with datasets:
    secgs    (N, 12, L)   float32   12-lead signals, standard lead order
    captions (N, max_len) int64     tokenized reference reports (start/end + pad)
    caplens  (N,)         int64     true caption length (before padding)

so that `TransformerDataset(out_dir, 'TEST')` loads it unchanged.

Reuses the SAME tokenizer as training (`encoder.pkl`) so the vocabulary matches.

These must match the PTB-XL preprocessing used for training, which is defined in
`configs/config_create_input_files.ini`:
      --sampling-rate 500   --length 5000   --max-len 64
      lead order = I, II, III, aVR, aVL, aVF, V1..V6
      report normalization = the same refinement applied to the PTB-XL targets
"""

import argparse
import os
import pickle

import numpy as np

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------------
# Signal loading
# ----------------------------------------------------------------------
def load_wfdb_signal(record_path, target_rate, target_len):
    """Read one WFDB record, reorder to STANDARD_LEADS, resample and crop/pad
    to (12, target_len). Returns float32 array or raises on failure."""
    import wfdb
    sig, fields = wfdb.rdsamp(record_path)          # sig: (T, n_leads)
    names = [n.strip() for n in fields["sig_name"]]
    fs = fields["fs"]

    # reorder leads to the standard 12-lead order (case-insensitive match)
    lower = {n.lower(): i for i, n in enumerate(names)}
    cols = []
    for lead in STANDARD_LEADS:
        idx = lower.get(lead.lower())
        if idx is None:
            raise ValueError(f"{record_path}: missing lead {lead} (has {names})")
        cols.append(sig[:, idx])
    x = np.stack(cols, axis=0).astype(np.float32)   # (12, T)

    # resample if needed
    if fs != target_rate:
        from scipy.signal import resample
        new_t = int(round(x.shape[1] * target_rate / fs))
        x = resample(x, new_t, axis=1).astype(np.float32)

    # crop / pad to target_len
    if x.shape[1] >= target_len:
        x = x[:, :target_len]
    else:
        x = np.pad(x, ((0, 0), (0, target_len - x.shape[1])), mode="constant")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ----------------------------------------------------------------------
# Report normalization (parity hook)
# ----------------------------------------------------------------------
def normalize_report(text):
    """Light normalization. For full parity with the PTB-XL targets, apply the
    same report refinement upstream and pass the refined text in here."""
    return " ".join(str(text).lower().split())


# ----------------------------------------------------------------------
# Tokenization (reuses training tokenizer)
# ----------------------------------------------------------------------
def encode_report(token_encoder, text, max_len):
    """Encode a report to a padded id array + true length, using the training
    tokenizer. Mirrors how the model encodes prompts/captions."""
    ids, length = token_encoder.encode(text, pad_to_length=max_len, add_special_tokens=True)
    ids = list(ids)[:max_len]
    if len(ids) < max_len:
        ids = ids + [token_encoder.vocab["<|pad|>"]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64), int(min(length, max_len))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build MIMIC-IV TEST hdf5 for cross-domain inference")
    ap.add_argument("--reports-csv", required=True,
                    help="CSV with columns: record_path (or path to WFDB record, no ext) and report text")
    ap.add_argument("--records-root", default="",
                    help="root dir prepended to record paths in the CSV (optional)")
    ap.add_argument("--record-col", default="record_path")
    ap.add_argument("--report-col", default="report")
    ap.add_argument("--encoder", required=True, help="path to training encoder.pkl (tokenizer)")
    ap.add_argument("--out-dir", required=True, help="output dir; writes TEST_DATASET.hdf5")
    ap.add_argument("--sampling-rate", type=int, default=500)
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--max-len", type=int, required=True, help="caption length (same as training)")
    ap.add_argument("--limit", type=int, default=-1, help="debug: cap number of records")
    args = ap.parse_args()

    import pandas as pd
    import h5py

    with open(args.encoder, "rb") as f:
        token_encoder = pickle.load(f)

    df = pd.read_csv(args.reports_csv)
    if args.limit > 0:
        df = df.iloc[:args.limit]

    secgs, captions, caplens, kept = [], [], [], 0
    skipped = 0
    for _, row in df.iterrows():
        rec = os.path.join(args.records_root, str(row[args.record_col])) if args.records_root else str(row[args.record_col])
        try:
            sig = load_wfdb_signal(rec, args.sampling_rate, args.length)
        except Exception as e:
            skipped += 1
            print(f"[skip] {rec}: {e}")
            continue
        text = normalize_report(row[args.report_col])
        ids, length = encode_report(token_encoder, text, args.max_len)
        secgs.append(sig)
        captions.append(ids)
        caplens.append(length)
        kept += 1

    if kept == 0:
        raise RuntimeError("No records were successfully prepared.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "TEST_DATASET.hdf5")
    with h5py.File(out_path, "w") as h:
        h.create_dataset("secgs", data=np.stack(secgs).astype(np.float32))
        h.create_dataset("captions", data=np.stack(captions).astype(np.int64))
        h.create_dataset("caplens", data=np.asarray(caplens, dtype=np.int64))

    print(f"\nWrote {out_path}: {kept} records ({skipped} skipped), "
          f"secgs shape=({kept}, 12, {args.length}), max_len={args.max_len}")
    print("Now run generation with the trained checkpoint pointing dataset_folder at this dir.")


if __name__ == "__main__":
    main()
