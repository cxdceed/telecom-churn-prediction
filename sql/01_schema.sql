-- SYNTHETIC DATA. SQLite-compatible schema (also works in PostgreSQL with minor type changes).
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS billing_history;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS churn_labels;

CREATE TABLE customers (
  customer_id TEXT PRIMARY KEY, signup_date TEXT, tenure_months INTEGER, age REAL,
  region TEXT, region_type TEXT, plan_type TEXT, contract_type TEXT, num_lines INTEGER,
  monthly_charge REAL, has_streaming_addon INTEGER, has_intl_addon INTEGER, autopay INTEGER,
  payment_method TEXT, paperless_billing INTEGER, device_financing_months_left INTEGER,
  competitor_promo_active INTEGER, recent_price_increase INTEGER);
CREATE TABLE billing_history (
  customer_id TEXT, billing_month TEXT, month_index INTEGER, plan_charge REAL, overage_charge REAL,
  data_gb_used REAL, paid_late INTEGER, days_late INTEGER, billed_amount REAL,
  PRIMARY KEY (customer_id, month_index));
CREATE TABLE support_tickets (
  ticket_id TEXT PRIMARY KEY, customer_id TEXT, created_date TEXT, category TEXT,
  resolved INTEGER, resolution_hours REAL, csat_score REAL);
CREATE TABLE churn_labels (customer_id TEXT PRIMARY KEY, churned_30d INTEGER, churn_date TEXT);
CREATE INDEX idx_billing_cust ON billing_history(customer_id);
CREATE INDEX idx_tickets_cust ON support_tickets(customer_id);
