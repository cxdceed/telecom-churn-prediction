"""Synthetic data generator for the telecom churn project.

!!! ALL DATA PRODUCED HERE IS SYNTHETIC. It encodes assumptions chosen by the
author, so any "finding" demonstrates methodology only, not real-world evidence.

Tables written to data/raw/:
    customers.csv        one row per customer (static attributes, deliberately messy)
    billing_history.csv  six monthly billing records per customer
    support_tickets.csv  one row per support ticket in the 180 days before the snapshot
    churn_labels.csv     target: did the customer cancel within 30 days after the snapshot

Planted (documented) effects:
    * month-to-month contracts, young tenure, late payments, competitor promos,
      recent price increases and bill shock raise churn risk; autopay,
      multiple lines, device financing and longer contracts lower it
    * ~55% of churners show a data-usage drop in the last 60 days (45% leave silently)
    * ~50% of churners open extra support tickets in the last 30 days
    * decoys: some non-churners also show usage drops / ticket bursts
    * overall churn rate calibrated to ~8%
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

PLANS = {"Basic": 25.0, "Standard": 40.0, "Premium": 55.0, "Unlimited": 70.0}
PLAN_USAGE_GB = {"Basic": 4.0, "Standard": 11.0, "Premium": 24.0, "Unlimited": 42.0}
PLAN_CAP_GB = {"Basic": 5.0, "Standard": 15.0, "Premium": 40.0, "Unlimited": np.inf}
REGIONS = {"Northeast": 0.25, "Southeast": 0.45, "Midwest": 0.20, "Southwest": 0.35, "West": 0.55}
N_BILL_MONTHS = 6
TICKET_WINDOW_DAYS = 180


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(z: np.ndarray, target: float) -> float:
    """Bisection search for the intercept giving mean(p) == target."""
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n: int = C.N_CUSTOMERS, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(C.AS_OF_DATE)

    # ---------------- static customer attributes ----------------
    customer_id = np.array([f"C{100000 + i}" for i in range(n)])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    is_m2m, is_2y = contract == "Month-to-month", contract == "Two year"

    gamma_params = {"Month-to-month": (1.4, 13), "One year": (2.0, 20), "Two year": (2.5, 22)}
    tenure = np.empty(n, dtype=int)
    for k, (shape, scale) in gamma_params.items():
        m = contract == k
        tenure[m] = np.clip(np.round(rng.gamma(shape, scale, m.sum())), 1, 96)
    signup_date = as_of - pd.to_timedelta(tenure * 30, unit="D")

    age = np.clip(np.round(rng.normal(44, 15, n)), 18, 85).astype(int)
    region = rng.choice(list(REGIONS), n, p=[0.22, 0.24, 0.20, 0.14, 0.20])
    region_type = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.45, 0.35, 0.20])
    plan = rng.choice(list(PLANS), n, p=[0.25, 0.35, 0.25, 0.15])
    lines = rng.choice([1, 2, 3, 4, 5], n, p=[0.38, 0.32, 0.15, 0.10, 0.05])

    base_price = np.array([PLANS[p] for p in plan])
    streaming = rng.random(n) < 0.30
    intl = rng.random(n) < 0.10
    monthly_charge = base_price * (1 + 0.55 * (lines - 1)) * np.where(is_2y, 0.95, 1.0)
    monthly_charge = monthly_charge + 10 * streaming + 8 * intl + rng.normal(0, 2.5, n)
    monthly_charge = np.round(np.clip(monthly_charge, 15, None), 2)

    autopay = rng.random(n) < np.where(is_m2m, 0.45, np.where(is_2y, 0.75, 0.62))
    pay_method = np.where(
        autopay,
        rng.choice(["Credit card", "Bank transfer", "Debit card"], n, p=[0.45, 0.35, 0.20]),
        rng.choice(["Credit card", "Bank transfer", "Debit card", "Mailed check"], n, p=[0.30, 0.20, 0.17, 0.33]),
    )
    paperless = rng.random(n) < 0.60
    financing_flag = rng.random(n) < 0.35
    fin_left = np.where(financing_flag, rng.integers(1, 25, n), 0)
    comp_promo = rng.random(n) < np.array([REGIONS[r] for r in region])
    price_increase = rng.random(n) < np.where(tenure > 36, 0.35, 0.18)

    # ---------------- late payments (observed before the snapshot) ----------------
    p_late = rng.beta(1.0, 12.0, n)
    p_late = p_late * np.where(autopay, 0.6, 1.6) * np.where(pay_method == "Mailed check", 2.0, 1.0)
    p_late = np.clip(p_late, 0, 0.7)
    late_flags = rng.random((n, N_BILL_MONTHS)) < p_late[:, None]
    late_count = late_flags.sum(1)

    # ---------------- baseline usage and ticket propensity ----------------
    plan_mean = np.array([PLAN_USAGE_GB[p] for p in plan])
    base_usage = rng.lognormal(np.log(plan_mean) - 0.125, 0.5) * np.sqrt(lines)
    cap = np.array([PLAN_CAP_GB[p] for p in plan])
    overage_prone = base_usage > cap
    frailty = rng.gamma(1.5, 1 / 1.5, n)  # mean-1 individual ticket propensity
    ticket_rate_6m = 0.12 * N_BILL_MONTHS * frailty * np.where(region_type == "Rural", 1.3, 1.0) \
        * np.where(price_increase, 1.4, 1.0)

    # ---------------- churn model (the data-generating process) ----------------
    z = (
        1.5 * is_m2m - 0.9 * is_2y
        - 0.03 * np.minimum(tenure, 60) + 0.6 * (tenure <= 6)
        + 0.020 * (monthly_charge - 60)
        + 0.35 * late_count
        + 0.50 * comp_promo
        + 0.45 * price_increase
        - 0.50 * autopay
        - 0.60 * financing_flag - 0.02 * fin_left
        + 0.30 * (frailty - 1)
        + 0.20 * (region_type == "Rural")
        + 0.25 * (age < 30)
        - 0.20 * (lines - 1)
        + 0.40 * overage_prone
        + rng.normal(0, 1.0, n)          # unobserved heterogeneity -> irreducible noise
    )
    z = z + _calibrate_intercept(z, C.TARGET_CHURN_RATE)
    p_churn = _sigmoid(z)
    churn = rng.random(n) < p_churn
    churn_date = as_of + pd.to_timedelta(rng.integers(1, 31, n), unit="D")

    # ---------------- billing history (6 months) ----------------
    usage_noise = np.clip(rng.normal(1.0, 0.10, (n, N_BILL_MONTHS)), 0.5, 1.6)
    factor = np.ones((n, N_BILL_MONTHS))
    signal = churn & (rng.random(n) < 0.55)               # 45% of churners leave silently
    decoy = (~churn) & (rng.random(n) < 0.20)             # non-churners with temporary dips
    drop_profile = np.array([1, 1, 1, 0.90, 0.72, 0.55])
    decoy_profile = np.array([1, 1, 1, 0.92, 0.75, 0.85])
    strength = rng.uniform(0.6, 1.4, n)[:, None]
    for mask, prof in ((signal, drop_profile), (decoy, decoy_profile)):
        f = np.clip(1 - (1 - prof)[None, :] * strength, 0.05, 1.1)
        factor[mask] = f[mask]
    usage = np.round(base_usage[:, None] * usage_noise * factor, 2)

    overage = np.clip((usage - cap[:, None]) * 10.0, 0, 60)
    overage = np.where(np.isfinite(cap)[:, None], overage, 0.0).round(2)
    plan_charge = np.repeat(monthly_charge[:, None], N_BILL_MONTHS, 1)
    plan_charge[price_increase, :4] = (monthly_charge[price_increase, None] / 1.07)
    plan_charge = plan_charge.round(2)
    days_late = np.where(late_flags, rng.poisson(6, (n, N_BILL_MONTHS)) + 1, 0)

    months = pd.period_range(end=as_of.to_period("M"), periods=N_BILL_MONTHS, freq="M")
    billing = pd.DataFrame({
        "customer_id": np.repeat(customer_id, N_BILL_MONTHS),
        "billing_month": np.tile(months.astype(str), n),
        "month_index": np.tile(np.arange(N_BILL_MONTHS), n),
        "plan_charge": plan_charge.ravel(),
        "overage_charge": overage.ravel(),
        "data_gb_used": usage.ravel(),
        "paid_late": late_flags.ravel().astype(int),
        "days_late": days_late.ravel(),
    })
    billing["billed_amount"] = (billing["plan_charge"] + billing["overage_charge"]).round(2)

    # ---------------- support tickets ----------------
    base_counts = rng.poisson(ticket_rate_6m)
    extra_counts = np.zeros(n, dtype=int)
    spike = churn & (rng.random(n) < 0.50)
    extra_counts[spike] = rng.poisson(0.8, spike.sum()) + 1
    decoy_t = (~churn) & (rng.random(n) < 0.12)
    extra_counts[decoy_t] = rng.poisson(1.0, decoy_t.sum()) + 1

    idx_b = np.repeat(np.arange(n), base_counts)
    idx_e = np.repeat(np.arange(n), extra_counts)
    age_days = np.concatenate([rng.integers(0, TICKET_WINDOW_DAYS, len(idx_b)), rng.integers(0, 30, len(idx_e))])
    is_extra = np.concatenate([np.zeros(len(idx_b), bool), np.ones(len(idx_e), bool)])
    idx = np.concatenate([idx_b, idx_e])
    churner_extra = is_extra & churn[idx]

    cats = np.array(["billing_dispute", "network_quality", "device_issue", "plan_change", "general_inquiry"])
    cat = np.where(
        churner_extra,
        rng.choice(cats, len(idx), p=[0.35, 0.40, 0.05, 0.15, 0.05]),
        rng.choice(cats, len(idx), p=[0.22, 0.30, 0.18, 0.15, 0.15]),
    )
    mult = np.where(cat == "network_quality", 1.5, np.where(cat == "billing_dispute", 0.8, 1.0))
    mult = mult * np.where(churner_extra, 1.4, 1.0)
    res_hours = np.round(rng.lognormal(np.log(18), 0.8, len(idx)) * mult, 1)
    p_res = np.where(churner_extra, 0.78, 0.95)
    p_res = np.where(age_days <= 2, np.minimum(p_res, 0.6), p_res)
    resolved = rng.random(len(idx)) < p_res
    csat_raw = 4.2 - 0.03 * res_hours - 0.5 * churner_extra + rng.normal(0, 0.9, len(idx))
    csat = np.clip(np.round(csat_raw), 1, 5)
    csat_present = resolved & (rng.random(len(idx)) < 0.60)

    tickets = pd.DataFrame({
        "customer_id": customer_id[idx],
        "created_date": (as_of - pd.to_timedelta(age_days, unit="D")).strftime("%Y-%m-%d"),
        "category": cat,
        "resolved": resolved.astype(int),
        "resolution_hours": np.where(resolved, res_hours, np.nan),
        "csat_score": np.where(csat_present, csat, np.nan),
    }).sort_values(["created_date", "customer_id"]).reset_index(drop=True)
    tickets.insert(0, "ticket_id", [f"T{500000 + i}" for i in range(len(tickets))])

    # ---------------- assemble customers + labels ----------------
    customers = pd.DataFrame({
        "customer_id": customer_id,
        "signup_date": signup_date.strftime("%Y-%m-%d"),
        "tenure_months": tenure,
        "age": age,
        "region": region,
        "region_type": region_type,
        "plan_type": plan,
        "contract_type": contract,
        "num_lines": lines,
        "monthly_charge": monthly_charge,
        "has_streaming_addon": streaming.astype(int),
        "has_intl_addon": intl.astype(int),
        "autopay": autopay.astype(int),
        "payment_method": pay_method,
        "paperless_billing": paperless.astype(int),
        "device_financing_months_left": fin_left,
        "competitor_promo_active": comp_promo.astype(int),
        "recent_price_increase": price_increase.astype(int),
    })
    labels = pd.DataFrame({
        "customer_id": customer_id,
        "churned_30d": churn.astype(int),
        "churn_date": np.where(churn, churn_date.strftime("%Y-%m-%d"), ""),
    })

    customers = _inject_messiness(customers, rng)
    return customers, billing, tickets, labels


def _inject_messiness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic data-quality problems so the cleaning step has real work to do."""
    df = df.copy()
    n = len(df)
    df.loc[rng.random(n) < 0.03, "age"] = np.nan                       # missing age
    bad = rng.random(n) < 0.003
    df.loc[bad, "age"] = rng.choice([-1, 0, 999], bad.sum())          # sentinel / impossible ages
    df["region_type"] = df["region_type"].astype(object)
    v = rng.random(n) < 0.015                                          # inconsistent casing / whitespace
    df.loc[v, "region_type"] = df.loc[v, "region_type"].map(lambda s: rng.choice([s.upper(), s.lower(), f" {s} "]))
    v = rng.random(n) < 0.01
    df.loc[v, "contract_type"] = df.loc[v, "contract_type"].map(
        lambda s: rng.choice([s.lower(), s.replace("-", " "), s.upper()]))
    df.loc[rng.random(n) < 0.01, "payment_method"] = np.nan            # missing payment method
    df.loc[rng.random(n) < 0.005, "monthly_charge"] = np.nan           # recoverable from billing_history
    dups = df.sample(frac=0.003, random_state=int(rng.integers(0, 1_000_000)))
    return pd.concat([df, dups], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC telecom churn data")
    ap.add_argument("--n-customers", type=int, default=C.N_CUSTOMERS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", default=str(C.DATA_RAW))
    a = ap.parse_args()

    customers, billing, tickets, labels = generate(a.n_customers, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out / "customers.csv", index=False)
    billing.to_csv(out / "billing_history.csv", index=False)
    tickets.to_csv(out / "support_tickets.csv", index=False)
    labels.to_csv(out / "churn_labels.csv", index=False)
    print(f"[SYNTHETIC DATA] customers={len(customers):,} (incl. duplicates) billing={len(billing):,} "
          f"tickets={len(tickets):,} churn_rate={labels['churned_30d'].mean():.2%}")


if __name__ == "__main__":
    main()
