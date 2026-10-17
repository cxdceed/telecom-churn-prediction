"""Retention economics: turn model scores into a contact list and a dollar figure.

Everything here uses ILLUSTRATIVE assumptions from src/config.py on SYNTHETIC data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def value_per_saved(monthly_charge, months=C.RETAINED_MONTHS, margin=C.GROSS_MARGIN):
    return np.asarray(monthly_charge) * months * margin


def net_value_curve(y, score, monthly_charge, save_rate=C.SAVE_RATE,
                    cost_per_contact=C.OFFER_COST + C.CONTACT_COST, n_points=200) -> pd.DataFrame:
    """Expected net value if we contact the top-x% of customers ranked by score."""
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind="mergesort")
    gain = (y[order] * save_rate * value_per_saved(np.asarray(monthly_charge)[order]))
    cum_gain, cum_cost = np.cumsum(gain), cost_per_contact * np.arange(1, len(y) + 1)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    return pd.DataFrame({
        "contact_fraction": (idx + 1) / len(y), "contacted": idx + 1,
        "churners_reached": np.cumsum(y[order])[idx], "gross_gain": cum_gain[idx],
        "cost": cum_cost[idx], "net_value": cum_gain[idx] - cum_cost[idx]})


def optimal_point(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["net_value"].idxmax()]


def value_aware_rule(y, prob, monthly_charge, save_rate=C.SAVE_RATE,
                     cost_per_contact=C.OFFER_COST + C.CONTACT_COST) -> dict:
    """Contact customer i iff  p_i * save_rate * value_i > cost  (uses calibrated probabilities)."""
    y, prob = np.asarray(y), np.asarray(prob)
    value = value_per_saved(monthly_charge)
    contact = prob * save_rate * value > cost_per_contact
    gain = (y[contact] * save_rate * value[contact]).sum()
    cost = cost_per_contact * contact.sum()
    return {"contacted": int(contact.sum()), "contact_fraction": float(contact.mean()),
            "net_value": float(gain - cost)}


def random_baseline(y, monthly_charge, fraction, save_rate=C.SAVE_RATE,
                    cost_per_contact=C.OFFER_COST + C.CONTACT_COST) -> float:
    y, mc = np.asarray(y), np.asarray(monthly_charge)
    k = int(round(fraction * len(y)))
    exp_gain = (y * save_rate * value_per_saved(mc)).mean() * k
    return float(exp_gain - cost_per_contact * k)


def sensitivity(y, score, monthly_charge, save_rates=(0.15, 0.30, 0.45), costs=(15, 28, 50)) -> pd.DataFrame:
    rows = []
    for sr in save_rates:
        for c in costs:
            best = optimal_point(net_value_curve(y, score, monthly_charge, sr, c))
            rows.append({"save_rate": sr, "cost_per_contact": c,
                         "optimal_contact_fraction": best.contact_fraction, "net_value": best.net_value})
    return pd.DataFrame(rows)
