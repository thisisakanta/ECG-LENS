"""
Shared utilities for ECG-LENS: training helpers, metrics, the report tokenizer,
and the PTB-XL data-preparation pipeline.

Everything a fresh clone needs to go from RAW PTB-XL to trained model is here:
`create_decoder_input_files` reads the raw records + reports, builds the
tokenizer (`encoder.pkl`) and the `TRAIN/VAL/TEST_DATASET.hdf5` files that the
rest of the pipeline consumes. Heavy dependencies (torch, wfdb, pandas, h5py)
are imported lazily inside the functions that use them.
"""

import csv
import math
import os
import pickle
import re
import shutil
from collections import Counter


# ======================================================================
# Training helpers
# ======================================================================
class AverageMeter:
    """Tracks the running average of a scalar (loss, timing, ...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def clip_gradient(optimizer, grad_clip):
    """Clip gradients in-place to [-grad_clip, grad_clip] for all params."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """Multiply every param-group learning rate by shrink_factor."""
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * shrink_factor


# ======================================================================
# Tokenization helpers
# ======================================================================
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def word_tokenize_caption(caption, exceptions=None):
    """Tokenize a report string into a list of tokens.

    Whitespace + punctuation aware. Any token in `exceptions` (e.g. special
    tokens like '<|unk|>') is preserved verbatim rather than split.
    """
    if caption is None:
        return []
    exceptions = set(exceptions or [])
    tokens = []
    for chunk in str(caption).split():
        if chunk in exceptions:
            tokens.append(chunk)
        else:
            tokens.extend(_TOKEN_RE.findall(chunk))
    return tokens


def word_tokenize_caption_list(captions, exceptions=None):
    """Apply `word_tokenize_caption` to a list of report strings."""
    return [word_tokenize_caption(c, exceptions) for c in captions]


# ======================================================================
# Metrics
# ======================================================================
def _rouge_1_single(ref_tokens, hyp_tokens):
    if not hyp_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    ref_c, hyp_c = Counter(ref_tokens), Counter(hyp_tokens)
    overlap = sum(min(hyp_c[t], ref_c[t]) for t in hyp_c)
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_rouge_1(references, hypotheses):
    """Corpus ROUGE-1 (precision / recall / F1).

    @param references: List[List[List[str]]] -- per sample a list of reference
                       token lists (first reference is used).
    @param hypotheses: List[List[str]]        -- per sample the hypothesis tokens.
    """
    ps, rs, fs = [], [], []
    for refs, hyp in zip(references, hypotheses):
        ref = refs[0] if refs and isinstance(refs[0], list) else refs
        p, r, f = _rouge_1_single(ref, hyp)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = max(1, len(ps))
    return {
        "rouge1_precision": sum(ps) / n,
        "rouge1_recall": sum(rs) / n,
        "rouge1_f1_score": sum(fs) / n,
    }


# ======================================================================
# Checkpointing / logging
# ======================================================================
def save_checkpoint(output_folder, model_name, epoch, epochs_since_improvement,
                    model, optimizer, encoder_config, decoder_config,
                    meteor, meteor_str, is_best, extra_state=None):
    """Save the latest checkpoint (and a BEST copy when is_best).

    The saved dict carries everything the resume path and the inference scripts
    read: model/optimizer state, epoch bookkeeping, and the encoder/decoder
    configs needed to rebuild the model.
    """
    import torch

    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "meteor": meteor,
    }
    if extra_state:
        state.update(extra_state)

    os.makedirs(output_folder, exist_ok=True)
    latest_path = os.path.join(output_folder, f"checkpoint_{model_name}.pth.tar")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(
            output_folder, f"BEST_checkpoint_{model_name}_METEOR-{meteor_str}.pth.tar"
        )
        shutil.copyfile(latest_path, best_path)


def save_results_to_csv(output_folder, epoch, train_losses, val_meteor_scores,
                        test_bleu4, test_bleu1, test_bleu2, test_rouge1, test_meteor):
    """Write per-epoch curves + final held-out test metrics to a CSV."""
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, "training_and_test_results.csv")

    test_rows = [
        ("Test BLEU-1", test_bleu1),
        ("Test BLEU-2", test_bleu2),
        ("Test BLEU-4", test_bleu4),
        ("Test ROUGE-1 P", test_rouge1.get("rouge1_precision")),
        ("Test ROUGE-1 R", test_rouge1.get("rouge1_recall")),
        ("Test ROUGE-1 F", test_rouge1.get("rouge1_f1_score")),
        ("Test METEOR", test_meteor),
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Epoch", "Train Loss", "Validation METEOR", "", "Test Metric", "Test Value"])
        n = max(len(train_losses), len(val_meteor_scores))
        for i in range(n):
            tl = train_losses[i] if i < len(train_losses) else ""
            vm = val_meteor_scores[i] if i < len(val_meteor_scores) else ""
            if i < len(test_rows):
                tm, tv = test_rows[i]
            else:
                tm, tv = "", ""
            w.writerow([i + 1, tl, vm, "", tm, tv])
    return path


# ======================================================================
# Report tokenizer (pickled to encoder.pkl by create_decoder_input_files)
# ======================================================================
SPECIAL_TOKENS = ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unk|>"]


class WordTokenizer:
    """Word-level tokenizer over the clinical report vocabulary.

    Interface consumed across the codebase:
      .vocab (token->id), .rev_word_map (id->token), .special_tokens (dict),
      .vocab_size, .encode(text, pad_to_length, add_special_tokens) -> (ids, len),
      .decode(ids, skip_special_tokens) -> str.
    """

    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.rev_word_map = {i: t for t, i in self.vocab.items()}
        self.special_tokens = {t: self.vocab[t] for t in SPECIAL_TOKENS if t in self.vocab}
        self.vocab_size = len(self.vocab)

    def encode(self, text, pad_to_length=None, add_special_tokens=True):
        unk = self.vocab["<|unk|>"]
        tokens = word_tokenize_caption(text, exceptions=list(self.special_tokens.keys()))
        ids = [self.vocab.get(t, unk) for t in tokens]
        if add_special_tokens:
            ids = [self.vocab["<|startoftext|>"]] + ids + [self.vocab["<|endoftext|>"]]
        length = len(ids)
        if pad_to_length is not None:
            if length < pad_to_length:
                ids = ids + [self.vocab["<|pad|>"]] * (pad_to_length - length)
            else:
                ids = ids[:pad_to_length]
                length = pad_to_length
        return ids, length

    def decode(self, ids, skip_special_tokens=True):
        special_ids = set(self.special_tokens.values())
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in special_ids:
                continue
            out.append(self.rev_word_map.get(i, "<|unk|>"))
        return " ".join(out)


def build_tokenizer(reports, min_word_freq, vocab_size):
    """Build a WordTokenizer from training reports."""
    counter = Counter()
    for r in reports:
        counter.update(word_tokenize_caption(r))
    words = [w for w, c in counter.most_common() if c >= min_word_freq]
    if vocab_size and vocab_size > 0:
        words = words[:vocab_size]
    vocab = {}
    for tok in SPECIAL_TOKENS:            # special tokens get the first, stable ids
        vocab[tok] = len(vocab)
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return WordTokenizer(vocab)


# ======================================================================
# Report normalization (offline, rule-based)
# ======================================================================
_TERM_MAP = [
    (r"\bleft type\b|\blinkstyp\b|\bllt\b", "left axis deviation"),
    (r"\bright type\b|\brechtstyp\b", "right axis deviation"),
    (r"\bposition type normal\b|\blagetyp normal\b", "normal axis"),
    (r"\bpremature atrial contractions?\b", "supraventricular extrasystoles"),
    (r"\bpremature ventricular contractions?\b", "ventricular extrasystoles"),
    (r"\bnon specific\b", "nonspecific"),
    (r"\binferior infarction\b", "inferior myocardial infarction"),
    (r"\banterior infarction\b", "anterior myocardial infarction"),
    (r"\blead_avl\b", "avl"),
    (r"\blead_avf\b", "avf"),
]


def refine_report(text, replace_abbr_text=True):
    """Deterministic, offline normalization of a report string.

    Applies the standing rule set (lead naming, clinical terminology, spacing)
    used when building the hdf5 inputs, so that data preparation is reproducible
    without any external service. The clinically focused report refinement
    described in the paper is a separate, upstream step: point `report_col` at a
    column of already-refined reports (`report_en`) to train on those targets.
    """
    s = str(text).lower().strip()
    if replace_abbr_text:
        for pat, repl in _TERM_MAP:
            s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ======================================================================
# PTB-XL data preparation: raw records + reports -> hdf5 + encoder.pkl
# ======================================================================
def create_decoder_input_files(data_folder, output_folder, sampling_rate,
                               replace_abbr_text, min_word_freq, vocab_size,
                               max_len, translate_comments, split_method,
                               debug_mode):
    """Build TRAIN/VAL/TEST_DATASET.hdf5 and encoder.pkl from raw PTB-XL.

    Reads the PTB-XL database csv + WFDB records under `data_folder`, applies
    report normalization, builds the tokenizer from the training split, and
    writes the hdf5 datasets (`secgs`, `captions`, `caplens`) plus the pickled
    tokenizer (`encoder.pkl`) into `output_folder`.

    Split (`split_method='strat_fold'`): folds < 9 = TRAIN, 9 = VAL, 10 = TEST,
    matching the label construction in my_datasets.TransformerDataset.
    """
    import wfdb
    import numpy as np
    import pandas as pd
    import h5py

    os.makedirs(output_folder, exist_ok=True)

    filtered = os.path.join(data_folder, "ptbxl_database_filtered.csv")
    fallback = os.path.join(data_folder, "ptbxl_database.csv")
    db_csv = filtered if os.path.exists(filtered) else fallback
    df = pd.read_csv(db_csv, index_col="ecg_id", engine="python", on_bad_lines="skip")

    filename_col = "filename_hr" if int(sampling_rate) >= 500 else "filename_lr"
    report_col = "report_en" if (translate_comments and "report_en" in df.columns) else "report"

    if split_method != "strat_fold":
        raise ValueError(f"Unsupported split_method '{split_method}' (only 'strat_fold').")
    splits = {
        "TRAIN": df[df.strat_fold < 9],
        "VAL": df[df.strat_fold == 9],
        "TEST": df[df.strat_fold == 10],
    }

    # Vocabulary is built from the TRAINING reports only.
    train_reports = [refine_report(r, replace_abbr_text) for r in splits["TRAIN"][report_col].fillna("")]
    tokenizer = build_tokenizer(train_reports, min_word_freq, vocab_size)
    with open(os.path.join(output_folder, "encoder.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[create_input] vocab size = {tokenizer.vocab_size} (from {len(train_reports)} train reports)")

    for split, sdf in splits.items():
        secgs, captions, caplens = [], [], []
        rows = list(sdf.iterrows())
        if debug_mode:
            rows = rows[:64]
        for _, row in rows:
            rec_path = os.path.join(data_folder, str(row[filename_col]))
            try:
                sig, _ = wfdb.rdsamp(rec_path)          # (T, 12)
            except Exception as e:
                print(f"[create_input] skip {rec_path}: {e}")
                continue
            x = np.nan_to_num(sig.T.astype(np.float32))  # (12, T)
            ids, length = tokenizer.encode(
                refine_report(row[report_col], replace_abbr_text),
                pad_to_length=max_len, add_special_tokens=True,
            )
            secgs.append(x)
            captions.append(np.asarray(ids, dtype=np.int64))
            caplens.append(int(length))

        out_path = os.path.join(output_folder, f"{split}_DATASET.hdf5")
        with h5py.File(out_path, "w") as h:
            h.create_dataset("secgs", data=np.stack(secgs).astype(np.float32))
            h.create_dataset("captions", data=np.stack(captions).astype(np.int64))
            h.create_dataset("caplens", data=np.asarray(caplens, dtype=np.int64))
        print(f"[create_input] wrote {out_path}: {len(secgs)} records")
