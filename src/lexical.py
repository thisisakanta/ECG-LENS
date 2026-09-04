"""
Canonical lexical metrics for ECG report generation (single source of truth).

Used for both PTB-XL (in-domain) and MIMIC-IV (cross-domain) evaluation so the
numbers are computed identically. Prefers NLTK (to match the training script's
corpus_bleu / meteor_score exactly) and falls back to dependency-free pure-Python
implementations when NLTK is unavailable.

Inputs are TOKENIZED reports:
    hypotheses : List[List[str]]                       (one token list per report)
    references : List[List[List[str]]]                 (per report, a list of refs)
This matches the format dumped by the training/inference pipeline
(`transformer_tokenized_hypotheses.pkl` / `transformer_tokenized_references.pkl`).
"""

import math
from collections import Counter
from statistics import mean

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except Exception:  # pragma: no cover - fallback path
    _HAS_NLTK = False


# ----------------------------------------------------------------------
# Pure-python fallbacks
# ----------------------------------------------------------------------
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def _bleu_n(hyp, ref, n):
    if not hyp or not ref:
        return 0.0
    weights = []
    for k in range(1, n + 1):
        h = _ngrams(hyp, k)
        r = _ngrams(ref, k)
        if not h:
            return 0.0
        overlap = sum(min(c, r[g]) for g, c in h.items())
        weights.append(overlap / sum(h.values()))
    if min(weights) == 0:
        p = 0.0
    else:
        p = math.exp(sum(math.log(w) for w in weights) / len(weights))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(1, len(hyp)))
    return bp * p


def _lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def _rouge_l(hyp, ref):
    if not hyp or not ref:
        return 0.0
    l = _lcs(hyp, ref)
    p, r = l / len(hyp), l / len(ref)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _rouge_1(hyp, ref):
    if not hyp or not ref:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    h, r = Counter(hyp), Counter(ref)
    overlap = sum(min(h[w], r[w]) for w in h)
    p = overlap / len(hyp)
    rec = overlap / len(ref)
    f = 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)
    return {"p": p, "r": rec, "f": f}


def _meteor_fallback(hyp, ref):
    if not hyp or not ref:
        return 0.0
    avail = Counter(ref)
    aligned = []
    for i, w in enumerate(hyp):
        if avail[w] > 0:
            aligned.append(i)
            avail[w] -= 1
    m = len(aligned)
    if m == 0:
        return 0.0
    P, R = m / len(hyp), m / len(ref)
    F = 10 * P * R / (R + 9 * P)
    chunks = 1
    for k in range(1, len(aligned)):
        if aligned[k] != aligned[k - 1] + 1:
            chunks += 1
    return F * (1 - 0.5 * ((chunks / m) ** 3))


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def compute_lexical_metrics(hypotheses, references):
    """Return a dict of corpus-level lexical metrics.

    @param hypotheses: List[List[str]]
    @param references: List[List[List[str]]]  (list of reference lists per hyp)
    """
    assert len(hypotheses) == len(references), "hyp/ref length mismatch"
    first_refs = [refs[0] for refs in references]  # single-reference setting

    if _HAS_NLTK:
        smooth = SmoothingFunction().method2
        third = 1.0 / 3.0
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu3 = corpus_bleu(references, hypotheses, weights=(third, third, third, 0), smoothing_function=smooth)
        bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smooth)
        meteor = mean(meteor_score(refs, hyp) for refs, hyp in zip(references, hypotheses))
    else:
        bleu1 = mean(_bleu_n(h, r, 1) for h, r in zip(hypotheses, first_refs))
        bleu2 = mean(_bleu_n(h, r, 2) for h, r in zip(hypotheses, first_refs))
        bleu3 = mean(_bleu_n(h, r, 3) for h, r in zip(hypotheses, first_refs))
        bleu4 = mean(_bleu_n(h, r, 4) for h, r in zip(hypotheses, first_refs))
        meteor = mean(_meteor_fallback(h, r) for h, r in zip(hypotheses, first_refs))

    r1 = [_rouge_1(h, r) for h, r in zip(hypotheses, first_refs)]
    rougeL = mean(_rouge_l(h, r) for h, r in zip(hypotheses, first_refs))

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "ROUGE-1-P": mean(x["p"] for x in r1),
        "ROUGE-1-R": mean(x["r"] for x in r1),
        "ROUGE-1-F": mean(x["f"] for x in r1),
        "ROUGE-L": rougeL,
        "METEOR": meteor,
        "n_samples": len(hypotheses),
        "backend": "nltk" if _HAS_NLTK else "pure-python",
    }


def per_sample_metrics(hypotheses, references):
    """Per-report BLEU-1/BLEU-4/ROUGE-L/METEOR (pure-python; for scatter plots
    and case-study tables)."""
    first_refs = [refs[0] for refs in references]
    rows = []
    for h, r in zip(hypotheses, first_refs):
        rows.append({
            "BLEU-1": _bleu_n(h, r, 1),
            "BLEU-3": _bleu_n(h, r, 3),
            "BLEU-4": _bleu_n(h, r, 4),
            "ROUGE-L": _rouge_l(h, r),
            "METEOR": _meteor_fallback(h, r),
        })
    return rows
