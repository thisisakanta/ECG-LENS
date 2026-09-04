"""
MERL-based zero-shot diagnostic label predictor.

This wraps the pretrained MERL (ECGCLIP) model so that ECG-LENS can build its
clinical prompt from *predicted* diagnostic labels instead of ground-truth
`scp_codes`. It reproduces MERL's zero-shot classification exactly:

    class_emb   = normalize(proj_t(text_encoder(prompt_c)))      # per class c
    ecg_emb     = normalize(MERL.ext_ecg_emb(ecg))
    logit_c     = ecg_emb @ class_emb                            # cosine sim
    score_c     = sigmoid((logit - mean) / std)                 # per family
    active      = { c : score_c > threshold }

Predictions are made per diagnostic family (superclass / subclass / form /
rhythm), each with its own class list, matching MERL's `zeroshot_val.py`.

NOTE (must confirm in your environment before running):
  * `MERL_REPO` below must point at the MERL-ICML2024 folder.
  * `checkpoint_path` must be the pretrained MERL (ECGCLIP) weights you used.
  * The ECG tensor fed to MERL must match MERL's expected input
    (12 leads, MERL's sampling/length). If ECG-LENS stores signals in a
    different shape/rate, adapt `_prepare_ecg` accordingly.
Nothing here is executed at import time; construct the class only when
`use_merl_labels` is enabled in the config.
"""

import os
import sys

import numpy as np
import torch

# Location of the MERL repo shipped in this project.
MERL_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MERL-ICML2024")


def _add_merl_to_path():
    for sub in ("utils", os.path.join("finetune", "models")):
        p = os.path.join(MERL_REPO, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


class MerlLabelPredictor:
    def __init__(
        self,
        checkpoint_path,
        prompt_json=None,
        ecg_model="vit_tiny",
        text_model="ncbi/MedCPT-Query-Encoder",
        proj_hidden=256,
        proj_out=256,
        num_leads=12,
        threshold=0.5,
        ensure_topk_super=1,
        batch_size=256,
        device=None,
    ):
        _add_merl_to_path()
        from utils_builder import ECGCLIP  # imported lazily; needs transformers + MERL repo

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.ensure_topk_super = ensure_topk_super
        self.batch_size = batch_size

        network_config = {
            "ecg_model": ecg_model,
            "num_leads": num_leads,
            "text_model": text_model,
            "projection_head": {"mlp_hidden_size": proj_hidden, "projection_size": proj_out},
        }
        self.model = ECGCLIP(network_config).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[MERL] missing keys: {len(missing)} (ok if only text-model buffers)")
        if unexpected:
            print(f"[MERL] unexpected keys: {len(unexpected)}")
        self.model.eval()

        self.prompt_dict = {}
        if prompt_json is None:
            prompt_json = os.path.join(MERL_REPO, "zeroshot", "CKEPE_prompt.json")
        if os.path.exists(prompt_json):
            import json
            with open(prompt_json, "r") as f:
                self.prompt_dict = json.load(f)

        self._weight_cache = {}  # family_key -> (codes, zeroshot_weights)

    # --- text side ------------------------------------------------------
    def _prompt_for_code(self, code, code_to_text=None):
        if code in self.prompt_dict:
            return self.prompt_dict[code].lower()
        if code_to_text is not None:
            txt = code_to_text(code)
            if txt:
                return str(txt).lower()
        return code.replace("_", " ").replace("/", " or ").lower()

    @torch.no_grad()
    def _class_weights(self, family_key, codes, code_to_text=None):
        if family_key in self._weight_cache:
            return self._weight_cache[family_key]
        cols = []
        for code in codes:
            text = self._prompt_for_code(code, code_to_text)
            tok = self.model._tokenize([text])
            emb = self.model.get_text_emb(
                tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
            )
            emb = self.model.proj_t(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            emb = emb / emb.norm()
            cols.append(emb)
        weights = torch.stack(cols, dim=1)  # (D, num_classes)
        self._weight_cache[family_key] = (list(codes), weights)
        return self._weight_cache[family_key]

    # --- ecg side -------------------------------------------------------
    def _prepare_ecg(self, secgs):
        """secgs: np.ndarray (N, C, L). Return float tensor MERL expects (N, 12, L).
        Adapt here if ECG-LENS stores a different lead layout / sampling rate."""
        x = np.asarray(secgs)
        if x.ndim == 2:
            x = x[:, None, :]
        return torch.as_tensor(x, dtype=torch.float32)

    @torch.no_grad()
    def _ecg_embeddings(self, secgs):
        ecg = self._prepare_ecg(secgs)
        embs = []
        for start in range(0, ecg.shape[0], self.batch_size):
            batch = ecg[start:start + self.batch_size].to(self.device)
            e = self.model.ext_ecg_emb(batch)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu())
        return torch.cat(embs, dim=0)  # (N, D)

    @staticmethod
    def _scores(ecg_emb, weights):
        logits = ecg_emb @ weights                       # (N, num_classes)
        norm = (logits - logits.mean()) / (logits.std() + 1e-8)
        return torch.sigmoid(norm)

    @torch.no_grad()
    def predict_active_codes(self, secgs, families, code_to_text=None):
        """
        families: dict family_key -> list of class codes, e.g.
          {"super": ["NORM","MI","STTC","CD","HYP"], "sub": [...],
           "form": [...], "rhythm": [...]}
        Returns: list (len N) of dicts { family_key: [active codes...] }.
        """
        ecg_emb = self._ecg_embeddings(secgs)
        n = ecg_emb.shape[0]
        out = [{k: [] for k in families} for _ in range(n)]
        for fam, codes in families.items():
            if not codes:
                continue
            _, weights = self._class_weights(fam, codes, code_to_text)
            scores = self._scores(ecg_emb, weights)       # (N, C)
            active = scores > self.threshold
            if fam == "super" and self.ensure_topk_super > 0:
                # never emit an empty superclass prompt
                topk = scores.topk(min(self.ensure_topk_super, scores.shape[1]), dim=1).indices
                for r in range(n):
                    active[r, topk[r]] = True
            codes_arr = list(codes)
            for r in range(n):
                out[r][fam] = [codes_arr[c] for c in range(len(codes_arr)) if active[r, c]]
        return out
