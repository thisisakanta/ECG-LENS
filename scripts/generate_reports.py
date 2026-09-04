"""
Generate reports with a TRAINED ECG-LENS checkpoint on any prepared test split
(used here for MIMIC-IV cross-domain; works for PTB-XL too). No training.

It reconstructs the model from the checkpoint's saved encoder/decoder configs and
reuses the training pipeline's `validate()` so generation, tokenization, and the
dumped pickles are IDENTICAL to the in-domain run. Outputs:
    <output_dir>/transformer_tokenized_hypotheses.pkl
    <output_dir>/transformer_tokenized_references.pkl
and prints corpus BLEU/ROUGE/METEOR. Feed the pickles to
`scripts/run_lexical.py` for the final metrics table.

Prompt conditioning
-------------------
If the checkpoint was trained with clinical-prompt conditioning
(`use_scp_label_conditioning=True`), each MIMIC record needs a prompt. Since
MIMIC has no PTB-XL scp_codes, pass MERL-predicted prompts via `--label-prompts`
(a pickle: List[str], one serialized prompt per record, aligned to the hdf5
order). Produce them with `merl_predictor.py` on the MIMIC ECGs. If the checkpoint
does not use prompts, omit this.

Run from the repository root so that `src/` and the config-relative paths resolve.
"""

import argparse
import os
import pickle
import sys

import torch

# Make the core source importable so we reuse the exact training modules.
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models import Encoder                                   # noqa: E402
from my_datasets import TransformerDataset                   # noqa: E402
from train_transformer_decoder import (                      # noqa: E402
    Model, validate, TrainingLogger, load_checkpoint_compat,
)


def build_model(ckpt, token_encoder, device):
    enc_cfg = ckpt["encoder_config"]
    dec_cfg = ckpt["decoder_config"]

    num_leads = enc_cfg.get("num_leads", 12)
    num_ecg_tokens = enc_cfg["num_ecg_tokens"]
    tokens_per_lead = max(1, num_ecg_tokens // (num_leads + 1))

    encoder = Encoder(
        num_blocks=enc_cfg["num_blocks"],
        encoded_ecg_size=enc_cfg.get("encoded_ecg_size", 1),
        kernel_size=enc_cfg["kernel_size"],
        stem_kernel_size=enc_cfg["stem_kernel_size"],
        in_chan=enc_cfg["in_chan"],
        num_leads=num_leads,
        use_lead_embeddings=enc_cfg.get("use_lead_embeddings", True),
        tokens_per_lead=tokens_per_lead,
    ).to(device)

    gpt_args = dict(
        n_layer=dec_cfg["n_layer"], n_head=dec_cfg["n_head"], n_embd=dec_cfg["n_embd"],
        block_size=dec_cfg["block_size"], bias=dec_cfg["bias"],
        vocab_size=dec_cfg["vocab_size"], dropout=dec_cfg["dropout"],
    )
    model = Model(
        gpt_args, encoder, token_encoder, num_ecg_tokens,
        use_scp_label_conditioning=enc_cfg.get("use_scp_label_conditioning", False),
        label_prompt_max_tokens=enc_cfg.get("label_prompt_max_tokens", 0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dec_cfg["block_size"], enc_cfg


def main():
    ap = argparse.ArgumentParser(description="Generate reports with a trained checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-folder", required=True, help="dir containing TEST_DATASET.hdf5")
    ap.add_argument("--encoder", required=True, help="training encoder.pkl (tokenizer)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-prompts", default=None,
                    help="pickle List[str] of MERL prompts (needed if the ckpt uses prompts)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=1, help="1 = greedy (matches training)")
    ap.add_argument("--start-token", action="store_true", default=True)
    ap.add_argument("--remove-start-token", action="store_true", default=False)
    ap.add_argument("--normalize-ds", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.encoder, "rb") as f:
        token_encoder = pickle.load(f)
    token_encoder.vocab_size = len(token_encoder.vocab)

    ckpt = load_checkpoint_compat(args.checkpoint, map_location=device)
    model, block_size, enc_cfg = build_model(ckpt, token_encoder, device)
    uses_prompt = enc_cfg.get("use_scp_label_conditioning", False)

    # Build the MIMIC test set. use_scp_labels=False avoids the PTB-XL scp_codes
    # path; prompts (if needed) are attached from --label-prompts.
    test_ds = TransformerDataset(args.dataset_folder, "TEST", use_scp_labels=False)

    if uses_prompt:
        if not args.label_prompts:
            raise ValueError("Checkpoint uses prompt conditioning; pass --label-prompts "
                             "(MERL-predicted prompt per record).")
        with open(args.label_prompts, "rb") as f:
            prompts = pickle.load(f)
        if len(prompts) != test_ds.secgs.shape[0]:
            raise ValueError(f"label-prompts ({len(prompts)}) != records ({test_ds.secgs.shape[0]})")
        test_ds.scp_label_texts = list(prompts)

    if test_ds.secgs.ndim == 2:
        import numpy as np
        test_ds.secgs = np.expand_dims(test_ds.secgs, 1)
    if args.normalize_ds:
        m = test_ds.secgs.mean(axis=(0, 2), keepdims=True)
        s = test_ds.secgs.std(axis=(0, 2), keepdims=True)
        test_ds.secgs = (test_ds.secgs - m) / s

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, pin_memory=True)

    logger = TrainingLogger(os.path.join(args.output_dir, "generate_trace.txt"), enabled=True)
    # validate() dumps transformer_tokenized_{hypotheses,references}.pkl into output_dir
    # and prints corpus BLEU/ROUGE-1/METEOR.
    import train_transformer_decoder as ttd
    ttd.print_freq = 10
    ttd.sanity_check = False
    ttd.debug_mode = False
    validate(
        val_loader=loader, model=model, token_encoder=token_encoder,
        block_size=block_size, top_k=args.top_k,
        start_token=args.start_token, remove_start_token=args.remove_start_token,
        logger=logger, epoch=-1, log_generation_steps=False,
        log_validation_max_samples=0, output_folder=args.output_dir,
    )
    logger.close()
    print(f"\nDone. Pickles in {args.output_dir}. "
          f"Run: python scripts/run_lexical.py --hyp "
          f"{args.output_dir}/transformer_tokenized_hypotheses.pkl "
          f"--ref {args.output_dir}/transformer_tokenized_references.pkl --tag MIMIC-IV")


if __name__ == "__main__":
    main()
