"""Load the cleaned tables into SQLite, run the feature-extraction SQL and the analysis queries."""
from __future__ import annotations

import sqlite3

import pandas as pd

from src import config as C
from src.clean import clean_all

SQL_DIR = C.ROOT / "sql"


def build_db(db_path, raw_dir=C.DATA_RAW):
    customers, billing, tickets, labels, _ = clean_all(raw_dir, write_report=False)
    tickets = tickets.assign(created_date=tickets["created_date"].dt.strftime("%Y-%m-%d"))
    con = sqlite3.connect(db_path)
    con.executescript((SQL_DIR / "01_schema.sql").read_text())
    for name, df in [("customers", customers), ("billing_history", billing),
                     ("support_tickets", tickets), ("churn_labels", labels)]:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
        df[cols].to_sql(name, con, if_exists="append", index=False)
    con.commit()
    return con


def sql_features(con, as_of=C.AS_OF_DATE) -> pd.DataFrame:
    return pd.read_sql((SQL_DIR / "02_feature_extraction.sql").read_text(), con, params={"as_of": as_of})


def main():
    db = C.ROOT / "data" / "processed" / "telecom.db"
    con = build_db(db)
    feats = sql_features(con)
    print(f"SQL feature table: {feats.shape}")
    for q in (SQL_DIR / "03_analysis_queries.sql").read_text().split(";"):
        if "SELECT" in q.upper():
            print(pd.read_sql(q, con).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
