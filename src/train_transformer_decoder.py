import ast
import configparser
import os
import pickle
import random
import shutil
import time
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

from GPT import GPT, GPTConfig
from models import Encoder
from my_datasets import TransformerDataset
from utils import (
    AverageMeter,
    adjust_learning_rate,
    clip_gradient,
    evaluate_rouge_1,
    save_checkpoint,
    save_results_to_csv,
    word_tokenize_caption,
    word_tokenize_caption_list,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_nltk_resource(resource_path, download_name=None):
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        if download_name is None:
            download_name = resource_path.split("/")[-1]
        try:
            return nltk.download(download_name, quiet=True)
        except Exception:
            return False


def safe_meteor_score(references, hypothesis):
    try:
        return meteor_score(references, hypothesis)
    except LookupError:
        wordnet_ready = _ensure_nltk_resource("corpora/wordnet", "wordnet")
        omw_ready = _ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
        if wordnet_ready and omw_ready:
            try:
                return meteor_score(references, hypothesis)
            except LookupError:
                pass

        reference_tokens = [token for reference in references for token in reference]
        hypothesis_tokens = list(hypothesis)
        if not reference_tokens or not hypothesis_tokens:
            return 0.0

        reference_set = set(reference_tokens)
        hypothesis_set = set(hypothesis_tokens)
        overlap = len(reference_set & hypothesis_set)
        if overlap == 0:
            return 0.0

        precision = overlap / len(hypothesis_set)
        recall = overlap / len(reference_set)
        if precision + recall == 0:
            return 0.0
        return (2.0 * precision * recall) / (precision + recall)


def load_checkpoint_compat(checkpoint_path, map_location):
    # Training checkpoints include optimizer/config metadata, so they must be
    # loaded with weights_only=False on PyTorch 2.6+.
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions that do not support
        # the weights_only argument.
        return torch.load(checkpoint_path, map_location=map_location)


class TrainingLogger:
    def __init__(self, filepath, enabled=True):
        self.filepath = filepath
        self.enabled = enabled
        self._fh = None
        if enabled:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            self._fh = open(filepath, "a", encoding="utf-8")

    def log(self, message):
        if not self.enabled:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"[{timestamp}] {message}\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Model(torch.nn.Module):
    def __init__(
        self,
        gpt_model_args,
        ecg_encoder,
        token_encoder,
        num_ecg_tokens,
        use_scp_label_conditioning=False,
        label_prompt_max_tokens=0,
    ):
        super().__init__()
        gptconf = GPTConfig(**gpt_model_args)
        self.token_encoder = token_encoder
        self.ecg_encoder = ecg_encoder
        self.num_ecg_tokens = num_ecg_tokens  # dynamic: depends on encoder tokens_per_lead
        self.use_scp_label_conditioning = use_scp_label_conditioning
        self.label_prompt_max_tokens = label_prompt_max_tokens
        self.num_condition_tokens = self.num_ecg_tokens + self.label_prompt_max_tokens
        if self.use_scp_label_conditioning and self.label_prompt_max_tokens < 1:
            raise ValueError("label_prompt_max_tokens must be >= 1 when use_scp_label_conditioning=True")

        self.decoder = GPT(
            gptconf,
            ecg_emb_dim=self.ecg_encoder.embedding_size,
            padding_idx=token_encoder.vocab["<|pad|>"],
        )

    def _encode_label_prompts(self, scp_label_texts, device):
        if not self.use_scp_label_conditioning or scp_label_texts is None:
            return None

        if isinstance(scp_label_texts, str):
            scp_label_texts = [scp_label_texts]

        encoded_labels = []
        for label_text in scp_label_texts:
            encoded_label, _ = self.token_encoder.encode(
                label_text,
                pad_to_length=self.label_prompt_max_tokens,
                add_special_tokens=False,
            )
            encoded_labels.append(encoded_label)

        return torch.tensor(encoded_labels, dtype=torch.long, device=device)

    def forward(self, model_input, targets=None, ecg_emb=None):
        if len(model_input) == 3:
            ecg, input_tokens, scp_labels = model_input
        else:
            ecg, input_tokens = model_input
            scp_labels = None

        if ecg_emb is None:
            ecg_emb = self.ecg_encoder(ecg)  # (B, K, D), K = num_ecg_tokens
        label_prefix_tokens = self._encode_label_prompts(scp_labels, ecg.device)

        logits = self.decoder(ecg_emb, input_tokens, prefix_idx=label_prefix_tokens)  # (B, K+L+T, V)
        if targets is not None:
            # Pad targets to align with all conditioning tokens.
            if (self.num_condition_tokens - 1) > 0:
                padding = (
                    torch.tensor([self.token_encoder.vocab["<|pad|>"]], dtype=torch.int)
                    .unsqueeze(1)
                    .repeat(input_tokens.size(0), self.num_condition_tokens - 1)
                    .to(device)
                )
                targets = torch.cat([padding, targets], dim=1)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.token_encoder.vocab["<|pad|>"],
            )
            return logits, targets, loss
        return logits, targets

    @torch.no_grad()
    def generate(
        self,
        model_input,
        max_new_tokens,
        eos_idx,
        temperature=1.0,
        top_k=None,
        log_writer=None,
        log_prefix="",
        token_encoder=None,
    ):
        if len(model_input) == 3:
            ecg, _, scp_labels = model_input
        else:
            ecg, _ = model_input
            scp_labels = None

        sentence_tokens = torch.tensor([], dtype=torch.int).to(ecg.device)
        ecg_emb = self.ecg_encoder(ecg.unsqueeze(0))
        label_prefix_tokens = self._encode_label_prompts(scp_labels, ecg.device)

        prefix_len = 0 if label_prefix_tokens is None else label_prefix_tokens.size(1)
        current_len = model_input[1].size(1)
        remaining_budget = self.decoder.config.block_size - prefix_len - current_len
        max_new_tokens = max(0, min(max_new_tokens, remaining_budget))

        for step in range(max_new_tokens):
            logits, _ = self(model_input, ecg_emb=ecg_emb)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if log_writer is not None:
                current_input = model_input[1].squeeze(0).tolist()
                next_token_id = int(idx_next[0][0].item())
                next_prob = float(probs[0, next_token_id].item())
                if token_encoder is not None and next_token_id in token_encoder.rev_word_map:
                    next_token_text = token_encoder.rev_word_map[next_token_id]
                else:
                    next_token_text = "<unk_id>"
                log_writer.log(
                    f"{log_prefix} step={step} input_ids={current_input} "
                    f"next_token_id={next_token_id} next_token_text={next_token_text} "
                    f"next_token_prob={next_prob:.6f}"
                )

            sentence_tokens = torch.cat((sentence_tokens, idx_next), dim=1)
            if idx_next[0][0] == eos_idx:
                return sentence_tokens
            model_input[1] = torch.cat((model_input[1], idx_next), dim=1)

        return sentence_tokens


def calculate_total_parameters(model):
    return model.ecg_encoder.get_num_params() + model.decoder.get_num_params()


def unpack_model_inputs(batch_inputs):
    if isinstance(batch_inputs, (tuple, list)) and len(batch_inputs) == 3:
        ecgs, captions, scp_labels = batch_inputs
    else:
        ecgs, captions = batch_inputs
        scp_labels = None
    return ecgs, captions, scp_labels


def log_trainable_parameters(model, optim, logger):
    logger.log("===== Trainable Parameters =====")
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_count += n
        if param.requires_grad:
            trainable_count += n
            logger.log(f"TRAINABLE name={name} shape={tuple(param.shape)} count={n}")
        else:
            logger.log(f"FROZEN name={name} shape={tuple(param.shape)} count={n}")
    logger.log(f"TOTAL_PARAMS={total_count} TRAINABLE_PARAMS={trainable_count}")
    for group_idx, group in enumerate(optim.param_groups):
        lr = group.get("lr", None)
        finetune_flag = group.get("finetune", None)
        logger.log(f"OPTIM_GROUP idx={group_idx} lr={lr} finetune={finetune_flag} n_params={len(group['params'])}")


def main():
    global_start = time.time()
    global print_freq, sanity_check, debug_mode

    config = configparser.ConfigParser()
    config.read(os.path.join("configs", "config_train_transformer_decoder.ini"))

    paths = config["file_names_and_paths"]
    data_folder = paths["dataset_folder"]
    model_name = paths["model_name"]
    output_folder = paths.get("output_folder", "./")
    os.makedirs(output_folder, exist_ok=True)

    # Encoder parameters
    encoder_blocks = config.getint("encoder_parameters", "num_blocks")
    kernel_size_str = config.get("encoder_parameters", "kernel_size")
    kernel_size = ast.literal_eval(kernel_size_str)
    stem_kernel_size = config.getint("encoder_parameters", "stem_kernel_size")
    ecg_input_length = config.getint("encoder_parameters", "encoded_ecg_size")
    ecg_tokens_per_lead = config.getint("encoder_parameters", "ecg_tokens_per_lead", fallback=1)

    # Decoder parameters
    dropout = config.getfloat("decoder_parameters", "dropout")
    start_token = config.getboolean("decoder_parameters", "start_token")
    n_layer = config.getint("decoder_parameters", "num_layers")
    n_head = config.getint("decoder_parameters", "num_heads")
    bias = config.getboolean("decoder_parameters", "bias")
    embed_dim = config.getint("decoder_parameters", "emb_dim")

    # Training parameters
    start_epoch = config.getint("training_parameters", "start_epoch")
    epochs = config.getint("training_parameters", "epochs")
    batch_size = config.getint("training_parameters", "batch_size")
    encoder_lr = config.getfloat("training_parameters", "encoder_lr")
    decoder_lr = config.getfloat("training_parameters", "decoder_lr")
    grad_clip = config.getfloat("training_parameters", "grad_clip")
    remove_start_token = config.getboolean("training_parameters", "remove_start_token")
    print_freq = config.getint("training_parameters", "print_freq")
    top_k = config.getint("training_parameters", "top_k")
    early_stopping_trigger = config.getint("training_parameters", "early_stopping_trigger")
    debug_mode = config.getboolean("training_parameters", "debug_mode")
    sanity_check = config.getboolean("training_parameters", "sanity_check")
    normalize_ds = config.getboolean("training_parameters", "normalize_ds")
    adjust_learning_rate_after = config.getint("training_parameters", "adjust_learning_rate_after")

    resume_checkpoint = config.get("training_parameters", "resume_checkpoint", fallback="").strip()
    auto_resume_latest = config.getboolean("training_parameters", "auto_resume_latest", fallback=True)
    checkpoint_every_epochs = config.getint("training_parameters", "checkpoint_every_epochs", fallback=1)

    log_filename = config.get(
        "training_parameters",
        "log_filename",
        fallback=f"{model_name}_train_trace.txt",
    )
    log_every_train_batch = config.getboolean("training_parameters", "log_every_train_batch", fallback=True)
    log_generation_steps = config.getboolean("training_parameters", "log_generation_steps", fallback=True)
    log_validation_max_samples = config.getint(
        "training_parameters",
        "log_validation_max_samples",
        fallback=-1,
    )
    use_scp_label_conditioning = config.getboolean(
        "training_parameters",
        "use_scp_label_conditioning",
        fallback=False,
    )

    logger = TrainingLogger(os.path.join(output_folder, log_filename), enabled=True)

    # Print and persist the contents of the config file
    for section in config.sections():
        logger.log(f"[{section}]")
        print(f"[{section}]")
        for key, value in config.items(section):
            logger.log(f"{key} = {value}")
            print(f"{key} = {value}")
        logger.log("")
        print()

    logger.log("********************************************************************")
    logger.log(f"output model name is: {model_name}")
    logger.log(f"device: {device}")
    logger.log("********************************************************************")
    print("********************************************************************")
    print("output model name is: " + model_name)
    print("********************************************************************")
    print(device)

    with open(os.path.join(data_folder, "encoder.pkl"), "rb") as f:
        token_encoder = pickle.load(f)
    # Keep vocab metadata consistent if encoder.pkl was edited post-creation.
    token_encoder.vocab_size = len(token_encoder.vocab)

    # Optionally build the clinical prompt from MERL-predicted labels
    # (predict-then-generate) instead of ground-truth scp_codes.
    merl_predictor = None
    use_merl_labels = config.getboolean("training_parameters", "use_merl_labels", fallback=False)
    if use_merl_labels:
        if not use_scp_label_conditioning:
            raise ValueError("use_merl_labels=True requires use_scp_label_conditioning=True")
        from merl_predictor import MerlLabelPredictor
        merl_predictor = MerlLabelPredictor(
            checkpoint_path=config.get("training_parameters", "merl_checkpoint"),
            prompt_json=config.get("training_parameters", "merl_prompt_json", fallback=None) or None,
            ecg_model=config.get("training_parameters", "merl_ecg_model", fallback="vit_tiny"),
            threshold=config.getfloat("training_parameters", "merl_threshold", fallback=0.5),
            device=str(device),
        )
        logger.log(f"MERL label prediction ENABLED (ckpt={config.get('training_parameters', 'merl_checkpoint')})")

    train_ds = TransformerDataset(data_folder, "TRAIN", use_scp_labels=use_scp_label_conditioning, merl_predictor=merl_predictor)
    val_ds = TransformerDataset(data_folder, "VAL", use_scp_labels=use_scp_label_conditioning, merl_predictor=merl_predictor)
    test_ds = TransformerDataset(data_folder, "TEST", use_scp_labels=use_scp_label_conditioning, merl_predictor=merl_predictor)

    label_prompt_max_tokens = 0
    if use_scp_label_conditioning:
        all_label_texts = list(train_ds.scp_label_texts) + list(val_ds.scp_label_texts) + list(test_ds.scp_label_texts)
        label_prompt_max_tokens = max(
            len(word_tokenize_caption(label_text, None))
            for label_text in all_label_texts
        )

    if train_ds.secgs.ndim == 2:
        train_ds.secgs = np.expand_dims(train_ds.secgs, 1)
        val_ds.secgs = np.expand_dims(val_ds.secgs, 1)
        test_ds.secgs = np.expand_dims(test_ds.secgs, 1)

    if normalize_ds:
        train_ds_mean = train_ds.secgs.mean(axis=(0, 2), keepdims=True)
        train_ds_std = train_ds.secgs.std(axis=(0, 2), keepdims=True)
        train_ds.secgs = (train_ds.secgs - train_ds_mean) / train_ds_std
        val_ds.secgs = (val_ds.secgs - train_ds_mean) / train_ds_std
        test_ds.secgs = (test_ds.secgs - train_ds_mean) / train_ds_std

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    in_chan = train_ds.secgs.shape[1]
    ecg_encoder = Encoder(
        num_blocks=encoder_blocks,
        encoded_ecg_size=ecg_input_length,  # Kept for backward compat
        kernel_size=kernel_size,
        stem_kernel_size=stem_kernel_size,
        in_chan=in_chan,
        num_leads=12,
        use_lead_embeddings=True,
        tokens_per_lead=ecg_tokens_per_lead,
    )
    ecg_encoder.to(device)

    # ECG encoder outputs num_leads * ecg_tokens_per_lead + global_tokens.
    num_ecg_tokens = ecg_encoder.num_ecg_tokens
    num_condition_tokens = num_ecg_tokens + label_prompt_max_tokens
    logger.log(
        f"ECG encoder tokens_per_lead={ecg_tokens_per_lead}, num_ecg_tokens={num_ecg_tokens}, "
        f"use_scp_label_conditioning={use_scp_label_conditioning}, "
        f"label_prompt_max_tokens={label_prompt_max_tokens}, num_condition_tokens={num_condition_tokens}"
    )
    if use_scp_label_conditioning:
        logger.log(f"SCP prompt examples: {getattr(train_ds, 'scp_label_texts', [])[:3]}")

    block_size = train_ds.captions.shape[1] + num_condition_tokens - 1
    gpt_model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=embed_dim,
        block_size=block_size,
        bias=bias,
        vocab_size=len(token_encoder.vocab),
        dropout=dropout,
    )

    model = Model(
        gpt_model_args,
        ecg_encoder,
        token_encoder,
        num_ecg_tokens,
        use_scp_label_conditioning=use_scp_label_conditioning,
        label_prompt_max_tokens=label_prompt_max_tokens,
    )
    model.to(device)

    total_params = calculate_total_parameters(model)
    logger.log(f"Total number of parameters: {total_params/1e6:.2f}M")
    print(f"Total number of parameters: {total_params/1e6:.2f}M")

    optim = torch.optim.Adam(
        params=[
            {
                "params": filter(lambda p: p.requires_grad, model.ecg_encoder.parameters()),
                "lr": encoder_lr,
                "finetune": True,
            },
            {
                "params": filter(lambda p: p.requires_grad, model.decoder.parameters()),
                "lr": decoder_lr,
                "finetune": True,
            },
        ]
    )

    epochs_since_improvement = 0
    best_bleu4 = 0
    best_bleu1 = 0
    best_bleu2 = 0
    best_rouge1 = {
        "rouge1_precision": 0,
        "rouge1_recall": 0,
        "rouge1_f1_score": 0,
    }
    best_meteor = -1
    meteor_str = "0"
    best_loss = AverageMeter()
    train_losses = []
    val_losses = []
    val_meteor_scores = []

    latest_checkpoint_path = os.path.join(output_folder, f"checkpoint_{model_name}.pth.tar")
    best_checkpoint_path = ""

    checkpoint_to_resume = ""
    if resume_checkpoint:
        checkpoint_to_resume = resume_checkpoint
    elif auto_resume_latest and os.path.exists(latest_checkpoint_path):
        checkpoint_to_resume = latest_checkpoint_path

    if checkpoint_to_resume:
        logger.log(f"Attempting resume from checkpoint: {checkpoint_to_resume}")
        ckpt = load_checkpoint_compat(checkpoint_to_resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim_state = ckpt.get("optimizer", None)
        if isinstance(optim_state, dict):
            optim.load_state_dict(optim_state)
        elif optim_state is not None and hasattr(optim_state, "state_dict"):
            optim.load_state_dict(optim_state.state_dict())
        start_epoch = ckpt.get("epoch", -1) + 1
        epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
        best_meteor = ckpt.get("best_meteor", ckpt.get("meteor", best_meteor))
        best_bleu4 = ckpt.get("best_bleu4", best_bleu4)
        best_bleu1 = ckpt.get("best_bleu1", best_bleu1)
        best_bleu2 = ckpt.get("best_bleu2", best_bleu2)
        best_rouge1 = ckpt.get("best_rouge1", best_rouge1)
        train_losses = ckpt.get("train_losses", train_losses)
        val_losses = ckpt.get("val_losses", val_losses)
        val_meteor_scores = ckpt.get("val_meteor_scores", val_meteor_scores)
        best_checkpoint_path = ckpt.get("best_checkpoint_path", "")
        meteor_str = str(round(best_meteor, 3))
        # Verify ECG token count matches
        resumed_num_tokens = ckpt.get("encoder_config", {}).get("num_condition_tokens", num_condition_tokens)
        if resumed_num_tokens != num_condition_tokens:
            logger.log(
                f"WARNING: Checkpoint has {resumed_num_tokens} conditioning tokens but config expects "
                f"{num_condition_tokens}."
            )
        logger.log(
            f"Resume successful: start_epoch={start_epoch} "
            f"epochs_since_improvement={epochs_since_improvement} best_meteor={best_meteor} "
            f"num_condition_tokens={num_condition_tokens}"
        )

    log_trainable_parameters(model, optim, logger)

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == early_stopping_trigger:
            logger.log(f"Early stopping triggered at epoch {epoch}")
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % adjust_learning_rate_after == 0:
            adjust_learning_rate(optim, 0.8)
            logger.log(f"Decayed learning rates at epoch {epoch}")

        logger.log(f"===== EPOCH {epoch} START =====")
        train_loss = train(
            train_loader=train_loader,
            model=model,
            optim=optim,
            epoch=epoch,
            grad_clip=grad_clip,
            start_token=start_token,
            remove_start_token=remove_start_token,
            logger=logger,
            log_every_train_batch=log_every_train_batch,
        )
        train_losses.append(train_loss.avg)

        recent_bleu4, recent_bleu1, recent_bleu2, recent_loss, recent_rouge1, recent_meteor = validate(
            val_loader=val_loader,
            model=model,
            token_encoder=token_encoder,
            block_size=block_size,
            top_k=top_k,
            start_token=start_token,
            remove_start_token=remove_start_token,
            logger=logger,
            epoch=epoch,
            log_generation_steps=log_generation_steps,
            log_validation_max_samples=log_validation_max_samples,
            output_folder=output_folder,
        )

        val_losses.append(recent_loss.avg)
        val_meteor_scores.append(recent_meteor)

        is_best = recent_meteor > best_meteor
        if not is_best:
            epochs_since_improvement += 1
            logger.log(f"Epochs since last improvement: {epochs_since_improvement}")
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            best_bleu4 = recent_bleu4
            best_bleu1 = recent_bleu1
            best_bleu2 = recent_bleu2
            best_loss = recent_loss
            best_rouge1 = recent_rouge1
            best_meteor = recent_meteor
            meteor_str = str(round(best_meteor, 3))
            best_checkpoint_path = os.path.join(
                output_folder,
                "BEST_checkpoint_" + model_name + "_METEOR-" + meteor_str + ".pth.tar",
            )
            logger.log(f"New best model at epoch {epoch}: meteor={best_meteor}")

        encoder_config_dict = dict(
            num_blocks=encoder_blocks,
            kernel_size=kernel_size,
            stem_kernel_size=stem_kernel_size,
            encoded_ecg_size=ecg_input_length,
            in_chan=in_chan,
            num_leads=12,
            use_lead_embeddings=True,
            num_ecg_tokens=num_ecg_tokens,
            num_condition_tokens=num_condition_tokens,
            use_scp_label_conditioning=use_scp_label_conditioning,
            label_prompt_max_tokens=label_prompt_max_tokens,
        )
        decoder_config_dict = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=embed_dim,
            block_size=block_size,
            bias=bias,
            vocab_size=token_encoder.vocab_size,
            dropout=dropout,
        )

        extra_state = {
            "best_meteor": best_meteor,
            "best_bleu4": best_bleu4,
            "best_bleu1": best_bleu1,
            "best_bleu2": best_bleu2,
            "best_rouge1": best_rouge1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_meteor_scores": val_meteor_scores,
            "best_checkpoint_path": best_checkpoint_path,
            "log_file": os.path.join(output_folder, log_filename),
        }

        save_checkpoint(
            output_folder,
            model_name,
            epoch,
            epochs_since_improvement,
            model,
            optim,
            encoder_config_dict,
            decoder_config_dict,
            recent_meteor,
            meteor_str,
            is_best,
            extra_state=extra_state,
        )

        if checkpoint_every_epochs > 0 and (epoch + 1) % checkpoint_every_epochs == 0:
            epoch_ckpt = os.path.join(output_folder, f"checkpoint_{model_name}_epoch_{epoch}.pth.tar")
            shutil.copy2(latest_checkpoint_path, epoch_ckpt)
            logger.log(f"Saved periodic checkpoint: {epoch_ckpt}")

        summary = (
            "Best Validation: \n"
            "BLEU1-Score: {bleu1:.4f}--\t"
            "BLEU2-Score: {bleu2:.4f}--\t"
            "BLEU4-Score: {bleu4:.4f}\n"
            "ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f}\n"
            "METEOR-Score: {meteor:.4f}\n"
            "Loss: {loss.val:.3f} ({loss.avg:.3f})\n"
        ).format(
            bleu1=best_bleu1,
            bleu4=best_bleu4,
            bleu2=best_bleu2,
            rouge1_p=best_rouge1["rouge1_precision"],
            rouge1_r=best_rouge1["rouge1_recall"],
            rouge1_f=best_rouge1["rouge1_f1_score"],
            meteor=best_meteor,
            loss=best_loss,
        )
        print(summary)
        logger.log(summary)

        if debug_mode:
            logger.log("Debug mode enabled, stopping after one epoch.")
            break

    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        selected_ckpt = best_checkpoint_path
    else:
        selected_ckpt = latest_checkpoint_path

    logger.log(f"Loading checkpoint for test evaluation: {selected_ckpt}")
    best_checkpoint = load_checkpoint_compat(selected_ckpt, map_location=device)
    model.load_state_dict(best_checkpoint["model"])

    print("Testing model:")
    test_bleu4, test_bleu1, test_bleu2, test_loss, test_rouge1, test_meteor = validate(
        val_loader=test_loader,
        model=model,
        token_encoder=token_encoder,
        block_size=block_size,
        top_k=top_k,
        start_token=start_token,
        remove_start_token=remove_start_token,
        logger=logger,
        epoch=-1,
        log_generation_steps=log_generation_steps,
        log_validation_max_samples=log_validation_max_samples,
        output_folder=output_folder,
    )

    test_summary = (
        "\n"
        "TEST BLEU4-Score: {bleu4:.4f}\n"
        "TEST BLEU1-Score: {bleu1:.4f}\n"
        "TEST BLEU2-Score: {bleu2:.4f}\n"
        "ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f}\n"
        "TEST METEOR-Score: {meteor:.4f}\n"
    ).format(
        bleu4=test_bleu4,
        bleu1=test_bleu1,
        bleu2=test_bleu2,
        rouge1_p=test_rouge1["rouge1_precision"],
        rouge1_r=test_rouge1["rouge1_recall"],
        rouge1_f=test_rouge1["rouge1_f1_score"],
        meteor=test_meteor,
    )
    print(test_summary)
    logger.log(test_summary)

    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        tested_name = os.path.join(
            output_folder,
            "BEST_checkpoint_" + model_name + "_TEST_METEOR-" + str(round(test_meteor, 3)) + ".pth.tar",
        )
        os.replace(best_checkpoint_path, tested_name)
        logger.log(f"Renamed best checkpoint after test to: {tested_name}")

    runtime = time.strftime("%H:%M:%S", time.gmtime((time.time() - global_start)))
    print("###entire training runtime:###")
    print(runtime)
    logger.log(f"###entire training runtime:### {runtime}")

    save_results_to_csv(
        output_folder,
        epoch,
        train_losses,
        val_meteor_scores,
        test_bleu4,
        test_bleu1,
        test_bleu2,
        test_rouge1,
        test_meteor,
    )

    logger.close()


def train(
    train_loader,
    model,
    optim,
    epoch,
    grad_clip,
    start_token,
    remove_start_token,
    logger,
    log_every_train_batch,
):
    model.train()
    model.ecg_encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    for i, (batch_inputs, targets, caplens) in enumerate(tqdm.tqdm(train_loader)):
        data_time.update(time.time() - start)
        ecgs, captions, scp_labels = unpack_model_inputs(batch_inputs)

        if sanity_check:
            ecgs.fill_(1)

        ecgs = ecgs.to(device)
        captions = captions.to(device)
        targets = targets.to(device)
        caplens = caplens.to(device)

        if start_token and remove_start_token:
            captions = captions[:, 1:]
            targets = targets[:, 1:]

        optim.zero_grad(set_to_none=True)
        model_inputs = (ecgs, captions, scp_labels) if scp_labels is not None else (ecgs, captions)
        logits, output_targets, loss = model(model_inputs, targets)
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optim, grad_clip)

        optim.step()

        losses.update(loss.item(), sum(caplens).item())
        batch_time.update(time.time() - start)

        if log_every_train_batch:
            logger.log(
                f"TRAIN epoch={epoch} batch={i} ecgs_shape={tuple(ecgs.shape)} "
                f"captions_shape={tuple(captions.shape)} targets_shape={tuple(targets.shape)} "
                f"scp_label_example={scp_labels[0] if scp_labels else None} "
                f"logits_shape={tuple(logits.shape)} loss={loss.item():.6f}"
            )
            if captions.size(0) > 0:
                logger.log(
                    f"TRAIN epoch={epoch} batch={i} sample0_input_ids={captions[0].detach().cpu().tolist()}"
                )
                logger.log(
                    f"TRAIN epoch={epoch} batch={i} sample0_target_ids={targets[0].detach().cpu().tolist()}"
                )
                logger.log(
                    "TRAIN epoch={} batch={} sample0_model_target_ids={}".format(
                        epoch,
                        i,
                        output_targets[0].detach().cpu().tolist(),
                    )
                )

        if i % print_freq == 0:
            print(
                "\nEpoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                    str(epoch),
                    str(i),
                    str(len(train_loader)),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
        start = time.time()
    return losses


def validate(
    val_loader,
    model,
    token_encoder,
    block_size,
    top_k,
    start_token,
    remove_start_token,
    logger,
    epoch,
    log_generation_steps,
    log_validation_max_samples,
    output_folder,
):
    model.eval()
    model.ecg_encoder.eval()

    losses = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()

    eos_idx = token_encoder.vocab["<|endoftext|>"]
    if start_token and not remove_start_token:
        sos_idx = token_encoder.vocab["<|startoftext|>"]
    hypotheses = []
    references_list = []

    with torch.no_grad():
        for i, (batch_inputs, targets, caplens) in tqdm.tqdm(enumerate(val_loader)):
            ecgs, captions, scp_labels = unpack_model_inputs(batch_inputs)

            if sanity_check:
                ecgs = torch.ones((len(captions), 1, 7663))

            ecgs = ecgs.to(device)
            captions = captions.to(device)
            targets = targets.to(device)

            if start_token and not remove_start_token:
                if scp_labels is not None:
                    input_samples = [
                        [ecgs[sample_idx], torch.tensor([sos_idx]).unsqueeze(1).to(device), scp_labels[sample_idx]]
                        for sample_idx in range(ecgs.size(0))
                    ]
                else:
                    input_samples = [[ecg, torch.tensor([sos_idx]).unsqueeze(1).to(device)] for ecg in ecgs]
                generation_block_size = block_size - 1
            else:
                if scp_labels is not None:
                    input_samples = [
                        [ecgs[sample_idx], torch.tensor([], dtype=torch.int).to(device), scp_labels[sample_idx]]
                        for sample_idx in range(ecgs.size(0))
                    ]
                else:
                    input_samples = [[ecg, torch.tensor([], dtype=torch.int).to(device)] for ecg in ecgs]
                generation_block_size = block_size

            y = []
            for sample_idx, sample in enumerate(input_samples):
                do_step_logging = log_generation_steps and (
                    log_validation_max_samples < 0 or sample_idx < log_validation_max_samples
                )
                prefix = f"GEN epoch={epoch} batch={i} sample={sample_idx}"
                generated = model.generate(
                    sample,
                    generation_block_size,
                    eos_idx,
                    top_k=top_k,
                    log_writer=logger if do_step_logging else None,
                    log_prefix=prefix,
                    token_encoder=token_encoder,
                ).squeeze(0)
                y.append(generated)

            model_inputs = (ecgs, captions, scp_labels) if scp_labels is not None else (ecgs, captions)
            logits, targets, loss = model(model_inputs, targets)
            losses.update(loss.item(), sum(caplens).item())
            batch_time.update(time.time() - start)
            start = time.time()

            logger.log(
                f"VALIDATE epoch={epoch} batch={i} ecgs_shape={tuple(ecgs.shape)} "
                f"captions_shape={tuple(captions.shape)} targets_shape={tuple(targets.shape)} "
                f"scp_label_example={scp_labels[0] if scp_labels else None} "
                f"logits_shape={tuple(logits.shape)} loss={loss.item():.6f}"
            )

            if i % print_freq == 0:
                print(
                    "\nValidation: [{0}/{1}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                    )
                )

            for caption in captions:
                cur_cap = []
                for token in caption:
                    if token != token_encoder.vocab["<|pad|>"]:
                        cur_cap.append(token)
                    else:
                        break
                references_list.append([token_encoder.decode(cur_cap, skip_special_tokens=True)])

            for caption in y:
                cur_cap = []
                for token in caption:
                    if token != token_encoder.vocab["<|pad|>"]:
                        cur_cap.append(token)
                    else:
                        break
                hypotheses.append(token_encoder.decode(cur_cap, skip_special_tokens=True))

        tokenized_hypotheses = word_tokenize_caption_list(
            hypotheses,
            list(token_encoder.special_tokens.keys()),
        )
        tokenized_references = [
            word_tokenize_caption_list(reference, list(token_encoder.special_tokens.keys()))
            for reference in references_list
        ]

        print("predictions example:")
        print(tokenized_hypotheses[0:5])
        hyp_path = os.path.join(output_folder, "transformer_tokenized_hypotheses.pkl")
        with open(hyp_path, "wb") as f:
            pickle.dump(tokenized_hypotheses, f)

        print("references:")
        print(tokenized_references[0:5])
        ref_path = os.path.join(output_folder, "transformer_tokenized_references.pkl")
        with open(ref_path, "wb") as f:
            pickle.dump(tokenized_references, f)

        logger.log(f"Saved tokenized hypotheses to {hyp_path}")
        logger.log(f"Saved tokenized references to {ref_path}")

        smoothing_function = SmoothingFunction().method2
        bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses, smoothing_function=smoothing_function)
        bleu1 = corpus_bleu(
            tokenized_references,
            tokenized_hypotheses,
            weights=[1, 0, 0, 0],
            smoothing_function=smoothing_function,
        )
        bleu2 = corpus_bleu(
            tokenized_references,
            tokenized_hypotheses,
            weights=[0.5, 0.5, 0, 0],
            smoothing_function=smoothing_function,
        )
        rouge1 = evaluate_rouge_1(tokenized_references, tokenized_hypotheses)
        meteor = mean(
            safe_meteor_score(cur_reference, cur_hypothesis)
            for cur_reference, cur_hypothesis in zip(tokenized_references, tokenized_hypotheses)
        )

        metrics = (
            "\n * BLEU-1 - {bleu1:.4f}, BLEU-2 - {bleu2:.4f}, BLEU-4 - {bleu4:.4f}, "
            "ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f} METEOR - {meteor:.4f}\n"
        ).format(
            bleu1=bleu1,
            bleu4=bleu4,
            bleu2=bleu2,
            rouge1_p=rouge1["rouge1_precision"],
            rouge1_r=rouge1["rouge1_recall"],
            rouge1_f=rouge1["rouge1_f1_score"],
            meteor=meteor,
        )
        print(metrics)
        logger.log(metrics)

    return bleu4, bleu1, bleu2, losses, rouge1, meteor


if __name__ == "__main__":
    random.seed(10)
    torch.manual_seed(10)
    np.random.seed(10)
    main()
