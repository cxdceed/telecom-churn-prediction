"""Streamlit demo: score a customer or a batch file. SYNTHETIC DATA / DEMO ONLY.

Run:  streamlit run app/streamlit_app.py      (requires `make all` first so models/ exists)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import business as B
from src import config as C

st.set_page_config(page_title="Churn Risk Scoring (synthetic demo)", layout="wide")
st.warning("All data and model outputs are SYNTHETIC. This demo illustrates methodology only.")


@st.cache_resource
def load():
    pipe = joblib.load(C.MODELS / "xgboost.joblib")
    iso = joblib.load(C.MODELS / "calibrator.joblib")
    ref = pd.read_csv(C.DATA_PROCESSED / "model_table.csv").drop(columns=[C.TARGET])
    return pipe, iso, ref


pipe, iso, ref = load()
defaults = {c: (ref[c].mode()[0] if ref[c].dtype == object else ref[c].median())
            for c in ref.columns if c != C.ID_COL}


def score(df):
    raw = pipe.predict_proba(df.drop(columns=[C.ID_COL], errors="ignore"))[:, 1]
    return iso.predict(raw)


tab1, tab2 = st.tabs(["Single customer", "Batch scoring"])
with tab1:
    c1, c2, c3 = st.columns(3)
    row = dict(defaults)
    row["contract_type"] = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    row["tenure_months"] = c1.slider("Tenure (months)", 1, 96, 12)
    row["monthly_charge"] = c1.slider("Monthly charge ($)", 15, 250, 65)
    row["usage_change_60d_pct"] = c2.slider("Data usage change, last 60d (%)", -80, 40, 0)
    row["tickets_30d"] = c2.slider("Support tickets, last 30d", 0, 6, 0)
    row["late_payments_6m"] = c2.slider("Late payments, last 6 mo", 0, 6, 0)
    row["autopay"] = int(c3.checkbox("Autopay enabled", True))
    row["competitor_promo_active"] = int(c3.checkbox("Competitor promo in region", False))
    row["device_financing_months_left"] = c3.slider("Device financing months left", 0, 24, 0)
    row["tickets_90d"] = max(row["tickets_30d"], int(defaults["tickets_90d"]))
    row["tickets_180d"] = max(row["tickets_90d"], int(defaults["tickets_180d"]))
    row["charge_per_line"] = row["monthly_charge"] / row["num_lines"]
    p = float(score(pd.DataFrame([row]))[0])
    val = float(B.value_per_saved([row["monthly_charge"]])[0])
    ev = p * C.SAVE_RATE * val - (C.OFFER_COST + C.CONTACT_COST)
    m1, m2, m3 = st.columns(3)
    m1.metric("Calibrated 30-day churn probability", f"{p:.1%}")
    m2.metric("Expected value of outreach", f"${ev:,.0f}")
    m3.metric("Recommendation", "Contact" if ev > 0 else "Do not contact")
    st.caption("Expected value = p x save-rate x (monthly charge x 12 x margin) - cost per contact. "
               "Assumptions live in src/config.py.")
with tab2:
    up = st.file_uploader("Upload CSV in model_table.csv format", type="csv")
    if up:
        d = pd.read_csv(up)
        d["churn_probability"] = score(d.drop(columns=[C.TARGET], errors="ignore"))
        d = d.sort_values("churn_probability", ascending=False)
        st.dataframe(d.head(200))
        st.download_button("Download scored file", d.to_csv(index=False), "scored_customers.csv")
