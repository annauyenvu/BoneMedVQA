"""Open-answer generation metrics (lightweight implementations)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _nltk_bleu(refs: list[str], hyps: list[str]) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothie = SmoothingFunction().method1
        scores = []
        for r, h in zip(refs, hyps):
            scores.append(
                sentence_bleu([_tokens(r)], _tokens(h), smoothing_function=smoothie)
            )
        return float(sum(scores) / max(len(scores), 1))
    except Exception:
        # Fallback overlap proxy
        scores = []
        for r, h in zip(refs, hyps):
            rt, ht = set(_tokens(r)), set(_tokens(h))
            scores.append(len(rt & ht) / max(len(rt | ht), 1))
        return float(sum(scores) / max(len(scores), 1))


def rouge_l(refs: list[str], hyps: list[str]) -> float:
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    scores = []
    for r, h in zip(refs, hyps):
        rt, ht = _tokens(r), _tokens(h)
        if not rt or not ht:
            scores.append(0.0)
            continue
        l = lcs(rt, ht)
        prec = l / len(ht)
        rec = l / len(rt)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append(2 * prec * rec / (prec + rec))
    return float(sum(scores) / max(len(scores), 1))


def exact_match_single(reference: str, hypothesis: str) -> float:
    return float(reference.strip().lower() == hypothesis.strip().lower())


def token_f1_single(reference: str, hypothesis: str) -> float:
    return token_f1([reference], [hypothesis])


def token_f1(refs: list[str], hyps: list[str]) -> float:
    scores = []
    for r, h in zip(refs, hyps):
        rc, hc = Counter(_tokens(r)), Counter(_tokens(h))
        overlap = sum((rc & hc).values())
        if overlap == 0:
            scores.append(0.0)
            continue
        prec = overlap / max(sum(hc.values()), 1)
        rec = overlap / max(sum(rc.values()), 1)
        scores.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float(sum(scores) / max(len(scores), 1))


def compute_generation_metrics(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    if not references:
        return {"bleu": 0.0, "rouge_l": 0.0, "token_f1": 0.0, "exact_match": 0.0, "n": 0}
    em = sum(int(r.strip().lower() == h.strip().lower()) for r, h in zip(references, hypotheses))
    out = {
        "bleu": _nltk_bleu(references, hypotheses),
        "rouge_l": rouge_l(references, hypotheses),
        "token_f1": token_f1(references, hypotheses),
        "exact_match": em / len(references),
        "n": len(references),
    }
    # Optional BERTScore if installed and network/models available
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(hypotheses, references, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean().item())
    except Exception:
        out["bertscore_f1"] = None
    return out
