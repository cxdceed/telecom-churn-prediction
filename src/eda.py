"""Exploratory data analysis: figures + hypothesis tests -> reports/eda_stats.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.plotting import PALETTE, plt, save, style


def rate_by(df, col):
    return df.groupby(col, observed=True)[C.TARGET].agg(rate="mean", n="size").reset_index()


def main():
    style()
    df = pd.read_csv(C.DATA_PROCESSED / "model_table.csv")
    base = df[C.TARGET].mean()
    out = {"n_customers": len(df), "overall_churn_rate": base}

    # --- Fig 1: structural drivers ---
    df["tenure_bucket"] = pd.cut(df.tenure_months, [0, 6, 12, 24, 48, 100],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49+"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, title in zip(ax, ["contract_type", "tenure_bucket", "late_payments_6m"],
                             ["Contract type", "Tenure (months)", "Late payments (last 6 months)"]):
        d = rate_by(df.assign(late_payments_6m=df.late_payments_6m.clip(upper=4)), col)
        a.bar(d[col].astype(str), d.rate * 100, color=PALETTE["primary"])
        a.axhline(base * 100, color=PALETTE["accent"], ls="--", lw=1.2, label=f"overall {base:.1%}")
        a.set_title(title); a.set_ylabel("30-day churn rate (%)"); a.legend()
        for i, r in d.reset_index(drop=True).iterrows():
            a.text(i, r.rate * 100 + 0.2, f"{r.rate:.1%}", ha="center", fontsize=8)
    fig.suptitle("Structural churn drivers (synthetic data)", y=1.02, fontweight="bold")
    save(fig, "eda_structural_drivers.png")

    # --- Fig 2: behavioural leading indicators ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = pd.cut(df.usage_change_60d_pct, [-100, -40, -20, -10, 0, 10, 200],
                  labels=["< -40%", "-40 to -20", "-20 to -10", "-10 to 0", "0 to +10", "> +10%"])
    d = rate_by(df.assign(b=bins), "b")
    ax[0].bar(d.b.astype(str), d.rate * 100, color=PALETTE["primary"])
    ax[0].set_title("Change in data usage (last 2 mo vs prior 3 mo)")
    ax[0].set_ylabel("30-day churn rate (%)"); ax[0].tick_params(axis="x", rotation=30)
    t = rate_by(df.assign(b=df.tickets_30d.clip(upper=4).astype(int).astype(str).replace("4", "4+")), "b")
    ax[1].bar(t.b, t.rate * 100, color=PALETTE["accent"])
    ax[1].set_title("Support tickets in last 30 days"); ax[1].set_xlabel("tickets")
    for a, dd in ((ax[0], d), (ax[1], t)):
        for i, r in dd.reset_index(drop=True).iterrows():
            a.text(i, r.rate * 100 + 0.2, f"{r.rate:.1%}", ha="center", fontsize=8)
    save(fig, "eda_behavioural_signals.png")

    # --- Fig 3: correlation with target ---
    num = df.select_dtypes("number").drop(columns=[C.TARGET])
    corr = num.apply(lambda s: s.fillna(s.median()).corr(df[C.TARGET])).sort_values()
    top = pd.concat([corr.head(8), corr.tail(8)])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index, top.values, color=[PALETTE["good"] if v < 0 else PALETTE["accent"] for v in top.values])
    ax.set_title("Correlation with churn (top 16 numeric features)"); ax.set_xlabel("Pearson r (point-biserial)")
    save(fig, "eda_feature_correlation.png")

    # --- statistical tests ---
    ct = pd.crosstab(df.contract_type, df[C.TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    out["chi2_contract_vs_churn"] = {"chi2": chi2, "dof": dof, "p_value": p,
                                     "cramers_v": float(np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))))}
    for col in ["usage_change_60d_pct", "tickets_30d", "monthly_charge", "tenure_months"]:
        a, b = df.loc[df[C.TARGET] == 1, col].dropna(), df.loc[df[C.TARGET] == 0, col].dropna()
        u, p = stats.mannwhitneyu(a, b)
        out[f"mannwhitney_{col}"] = {"median_churn": float(a.median()), "median_retained": float(b.median()),
                                     "p_value": float(p),
                                     "rank_biserial": float(1 - 2 * u / (len(a) * len(b)))}
    out["churn_rate_by_contract"] = df.groupby("contract_type")[C.TARGET].mean().round(4).to_dict()
    out["churn_rate_by_tickets_30d"] = df.groupby(df.tickets_30d.clip(upper=3))[C.TARGET].mean().round(4).to_dict()
    out["churn_rate_usage_drop_over20pct"] = float(df.loc[df.usage_change_60d_pct < -20, C.TARGET].mean())
    out["share_of_customers_usage_drop_over20pct"] = float((df.usage_change_60d_pct < -20).mean())
    out["churners_without_usage_drop_or_ticket_spike"] = float(
        ((df.usage_change_60d_pct >= -20) & (df.tickets_30d == 0))[df[C.TARGET] == 1].mean())
    (C.REPORTS / "eda_stats.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
