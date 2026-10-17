"""Central configuration: paths, seeds and business assumptions.

ALL DATA IN THIS PROJECT IS SYNTHETIC. Nothing here describes a real carrier.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
N_CUSTOMERS = 50_000
AS_OF_DATE = "2025-12-31"        # snapshot date; label = cancels in the 30 days after
TARGET_CHURN_RATE = 0.08
TARGET = "churned_30d"
ID_COL = "customer_id"

# --- Business assumptions for the retention cost model (all illustrative) ---
OFFER_COST = 25.0          # average cost of a retention offer (credit/discount) per contacted customer
CONTACT_COST = 3.0         # outreach cost (agent time / SMS / campaign) per contacted customer
SAVE_RATE = 0.30           # share of contacted would-be churners who accept and stay
RETAINED_MONTHS = 12       # months of revenue assumed retained for a saved customer
GROSS_MARGIN = 0.40        # margin applied to retained revenue
