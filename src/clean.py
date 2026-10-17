"""Data cleaning: raw CSVs -> validated tables + a cleaning report.

Every rule is logged so the cleaning step is auditable (reports/cleaning_report.json).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C

CONTRACT_MAP = {"month-to-month": "Month-to-month", "month to month": "Month-to-month",
                "one year": "One year", "two year": "Two year"}


def load_raw(raw_dir=C.DATA_RAW):
    customers = pd.read_csv(raw_dir / "customers.csv")
    billing = pd.read_csv(raw_dir / "billing_history.csv")
    tickets = pd.read_csv(raw_dir / "support_tickets.csv")
    labels = pd.read_csv(raw_dir / "churn_labels.csv")
    return customers, billing, tickets, labels


def clean_customers(df: pd.DataFrame, billing: pd.DataFrame):
    rep = {"rows_in": len(df)}
    df = df.copy()

    # 1. exact duplicate customers (keep first)
    before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    rep["duplicate_rows_removed"] = before - len(df)

    # 2. normalise categorical text (casing / whitespace / separators)
    df["region_type"] = df["region_type"].str.strip().str.title()
    key = df["contract_type"].str.strip().str.lower()
    rep["contract_labels_standardised"] = int((df["contract_type"] != key.map(CONTRACT_MAP)).sum())
    df["contract_type"] = key.map(CONTRACT_MAP)

    # 3. impossible ages -> NaN (imputed later inside the modelling pipeline to avoid leakage)
    bad_age = (df["age"] < 18) | (df["age"] > 100)
    rep["impossible_ages_set_to_nan"] = int(bad_age.sum())
    df.loc[bad_age, "age"] = np.nan
    rep["age_missing_total"] = int(df["age"].isna().sum())

    # 4. missing monthly_charge -> recover from latest billed plan_charge
    latest = billing.sort_values("month_index").groupby("customer_id")["plan_charge"].last()
    miss = df["monthly_charge"].isna()
    rep["monthly_charge_recovered_from_billing"] = int(miss.sum())
    df.loc[miss, "monthly_charge"] = df.loc[miss, "customer_id"].map(latest)

    # 5. missing payment method -> explicit "Unknown" category (missingness may carry signal)
    rep["payment_method_missing_to_unknown"] = int(df["payment_method"].isna().sum())
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # 6. validation checks
    assert df["customer_id"].is_unique
    assert df["contract_type"].notna().all()
    assert df["monthly_charge"].notna().all()
    rep["rows_out"] = len(df)
    return df.reset_index(drop=True), rep


def clean_all(raw_dir=C.DATA_RAW, write_report: bool = True):
    customers, billing, tickets, labels = load_raw(raw_dir)
    customers, rep = clean_customers(customers, billing)
    billing = billing.drop_duplicates(["customer_id", "month_index"])
    tickets["created_date"] = pd.to_datetime(tickets["created_date"])
    rep["billing_rows"] = len(billing)
    rep["ticket_rows"] = len(tickets)
    rep["label_rows"] = len(labels)
    if write_report:
        C.REPORTS.mkdir(exist_ok=True, parents=True)
        (C.REPORTS / "cleaning_report.json").write_text(json.dumps(rep, indent=2))
    return customers, billing, tickets, labels, rep


if __name__ == "__main__":
    *_, report = clean_all()
    print(json.dumps(report, indent=2))
