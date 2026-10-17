"""Feature engineering: customer + billing + ticket tables -> one modelling row per customer.

All features use information available on or before the snapshot date, so there is no
target leakage. The SQL twin of this logic lives in sql/02_feature_extraction.sql.
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.clean import clean_all

CATEGORICAL = ["region", "region_type", "plan_type", "contract_type", "payment_method"]


def billing_features(billing: pd.DataFrame) -> pd.DataFrame:
    g = billing.groupby("customer_id")
    base = billing[billing["month_index"] <= 2].groupby("customer_id")["data_gb_used"].mean()
    recent = billing[billing["month_index"] >= 4].groupby("customer_id")["data_gb_used"].mean()
    first_charge = billing[billing["month_index"] == 0].set_index("customer_id")["plan_charge"]
    last_charge = billing[billing["month_index"] == 5].set_index("customer_id")["plan_charge"]
    out = pd.DataFrame({
        "late_payments_6m": g["paid_late"].sum(),
        "avg_days_late": g["days_late"].mean(),
        "avg_overage_6m": g["overage_charge"].mean(),
        "months_with_overage": g["overage_charge"].apply(lambda s: (s > 0).sum()),
        "data_gb_avg_6m": g["data_gb_used"].mean(),
        "data_gb_volatility": g["data_gb_used"].std() / g["data_gb_used"].mean(),
        "data_gb_baseline_3m": base,
        "data_gb_recent_2m": recent,
    })
    out["usage_change_60d_pct"] = (out["data_gb_recent_2m"] / out["data_gb_baseline_3m"] - 1) * 100
    out["charge_change_6m_pct"] = (last_charge / first_charge - 1) * 100
    return out


def ticket_features(tickets: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    t = tickets.copy()
    t["days_ago"] = (as_of - t["created_date"]).dt.days
    g = t.groupby("customer_id")
    out = pd.DataFrame({
        "tickets_180d": g.size(),
        "tickets_90d": t[t.days_ago <= 90].groupby("customer_id").size(),
        "tickets_30d": t[t.days_ago <= 30].groupby("customer_id").size(),
        "billing_tickets_90d": t[(t.days_ago <= 90) & (t.category == "billing_dispute")].groupby("customer_id").size(),
        "network_tickets_90d": t[(t.days_ago <= 90) & (t.category == "network_quality")].groupby("customer_id").size(),
        "unresolved_tickets": g["resolved"].apply(lambda s: (s == 0).sum()),
        "avg_resolution_hours": g["resolution_hours"].mean(),
        "avg_csat": g["csat_score"].mean(),
        "days_since_last_ticket": g["days_ago"].min(),
    })
    return out


def build_model_table(customers, billing, tickets, labels=None, as_of: str = C.AS_OF_DATE) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    df = customers.set_index("customer_id")
    df = df.join(billing_features(billing)).join(ticket_features(tickets, as_of))
    count_cols = ["tickets_180d", "tickets_90d", "tickets_30d", "billing_tickets_90d",
                  "network_tickets_90d", "unresolved_tickets"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)   # no tickets -> zero, not unknown
    df["charge_per_line"] = df["monthly_charge"] / df["num_lines"]
    df["ticket_spike_ratio"] = df["tickets_30d"] / (df["tickets_180d"] - df["tickets_30d"] + 1) * 5
    df = df.drop(columns=["signup_date"])
    if labels is not None:
        df = df.join(labels.set_index("customer_id")[[C.TARGET]])
    return df.reset_index()


def main():
    customers, billing, tickets, labels, _ = clean_all()
    mt = build_model_table(customers, billing, tickets, labels)
    C.DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
    mt.to_csv(C.DATA_PROCESSED / "model_table.csv", index=False)
    print(f"model_table: {mt.shape[0]:,} rows x {mt.shape[1]} cols | churn rate {mt[C.TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
