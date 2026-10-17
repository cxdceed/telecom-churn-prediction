"""Ranking-oriented metrics. Plain accuracy is useless at an 8% base rate."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_stats(y_true, score, frac: float = 0.10) -> dict:
    y_true, score = np.asarray(y_true), np.asarray(score)
    k = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-score, kind="mergesort")[:k]
    captured = y_true[top].sum()
    base = y_true.mean()
    return {
        f"recall_at_top{int(frac*100)}": captured / y_true.sum(),
        f"precision_at_top{int(frac*100)}": captured / k,
        f"lift_at_top{int(frac*100)}": (captured / k) / base,
    }


def summarise(y_true, score, prob_for_brier=None) -> dict:
    out = {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "baseline_pr_auc": float(np.mean(y_true)),
        **top_k_stats(y_true, score, 0.10),
        **top_k_stats(y_true, score, 0.05),
    }
    if prob_for_brier is not None:
        out["brier"] = brier_score_loss(y_true, prob_for_brier)
    return {k: float(v) for k, v in out.items()}


def gains_curve(y_true, score, n_points: int = 100):
    """Cumulative share of churners captured as we contact the top x% of customers."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    order = np.argsort(-score, kind="mergesort")
    cum = np.cumsum(y_true[order]) / y_true.sum()
    x = np.arange(1, len(y_true) + 1) / len(y_true)
    idx = np.linspace(0, len(y_true) - 1, n_points).astype(int)
    return x[idx], cum[idx]
