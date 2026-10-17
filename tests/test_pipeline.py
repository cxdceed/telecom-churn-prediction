"""Smoke + integrity tests on a small synthetic dataset (fast enough for CI)."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.build_sqlite import build_db, sql_features
from src.clean import clean_all
from src.features import build_model_table
from src.generate_data import generate


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    customers, billing, tickets, labels = generate(n=3000, seed=1)
    customers.to_csv(d / "customers.csv", index=False)
    billing.to_csv(d / "billing_history.csv", index=False)
    tickets.to_csv(d / "support_tickets.csv", index=False)
    labels.to_csv(d / "churn_labels.csv", index=False)
    return d


def test_generation_is_reproducible():
    a = generate(n=500, seed=5)[3]
    b = generate(n=500, seed=5)[3]
    assert a.equals(b)


def test_churn_rate_close_to_target(raw_dir):
    labels = pd.read_csv(raw_dir / "churn_labels.csv")
    assert 0.05 < labels[C.TARGET].mean() < 0.11


def test_target_not_in_customer_table(raw_dir):
    assert C.TARGET not in pd.read_csv(raw_dir / "customers.csv").columns


def test_cleaning_removes_problems(raw_dir):
    customers, billing, tickets, labels, rep = clean_all(raw_dir, write_report=False)
    assert customers["customer_id"].is_unique and len(customers) == 3000
    assert set(customers["contract_type"]) <= {"Month-to-month", "One year", "Two year"}
    assert customers["age"].dropna().between(18, 100).all()
    assert customers["monthly_charge"].notna().all()
    assert rep["duplicate_rows_removed"] > 0


def test_model_table_shape_and_no_leakage(raw_dir):
    customers, billing, tickets, labels, _ = clean_all(raw_dir, write_report=False)
    mt = build_model_table(customers, billing, tickets, labels)
    assert len(mt) == 3000 and mt[C.TARGET].isin([0, 1]).all()
    leaky = {"churn_date", "churned_30d_pred"}
    assert leaky.isdisjoint(mt.columns)
    assert (mt["tickets_30d"] <= mt["tickets_90d"]).all() and (mt["tickets_90d"] <= mt["tickets_180d"]).all()


def test_sql_features_match_pandas(raw_dir, tmp_path):
    customers, billing, tickets, labels, _ = clean_all(raw_dir, write_report=False)
    mt = build_model_table(customers, billing, tickets, labels).set_index("customer_id")
    con = build_db(tmp_path / "t.db", raw_dir)
    sq = sql_features(con).set_index("customer_id").loc[mt.index]
    for col in ["late_payments_6m", "tickets_30d", "tickets_90d", "tickets_180d", "unresolved_tickets"]:
        assert np.array_equal(sq[col].values, mt[col].values), col
    assert np.allclose(sq["usage_change_60d_pct"], mt["usage_change_60d_pct"], equal_nan=True)
