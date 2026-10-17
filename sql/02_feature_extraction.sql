-- SYNTHETIC DATA. Feature extraction: one row per customer, point-in-time as of :as_of.
-- Mirrors src/features.py (parity is checked in tests/test_sql_parity.py).
WITH bill AS (
  SELECT customer_id,
         SUM(paid_late)                                              AS late_payments_6m,
         AVG(days_late)                                              AS avg_days_late,
         AVG(overage_charge)                                         AS avg_overage_6m,
         AVG(data_gb_used)                                           AS data_gb_avg_6m,
         AVG(CASE WHEN month_index <= 2 THEN data_gb_used END)       AS data_gb_baseline_3m,
         AVG(CASE WHEN month_index >= 4 THEN data_gb_used END)       AS data_gb_recent_2m
  FROM billing_history
  GROUP BY customer_id
),
tix AS (
  SELECT customer_id,
         COUNT(*)                                                                            AS tickets_180d,
         SUM(CASE WHEN julianday(:as_of) - julianday(created_date) <= 90 THEN 1 ELSE 0 END) AS tickets_90d,
         SUM(CASE WHEN julianday(:as_of) - julianday(created_date) <= 30 THEN 1 ELSE 0 END) AS tickets_30d,
         SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END)                                       AS unresolved_tickets,
         AVG(resolution_hours)                                                               AS avg_resolution_hours,
         AVG(csat_score)                                                                     AS avg_csat
  FROM support_tickets
  GROUP BY customer_id
)
SELECT c.customer_id, c.tenure_months, c.contract_type, c.plan_type, c.monthly_charge,
       b.late_payments_6m, b.avg_days_late, b.avg_overage_6m,
       (b.data_gb_recent_2m / b.data_gb_baseline_3m - 1) * 100 AS usage_change_60d_pct,
       COALESCE(t.tickets_180d, 0)       AS tickets_180d,
       COALESCE(t.tickets_90d, 0)        AS tickets_90d,
       COALESCE(t.tickets_30d, 0)        AS tickets_30d,
       COALESCE(t.unresolved_tickets, 0) AS unresolved_tickets,
       t.avg_resolution_hours, t.avg_csat,
       l.churned_30d
FROM customers c
JOIN bill b           ON b.customer_id = c.customer_id
LEFT JOIN tix t       ON t.customer_id = c.customer_id
LEFT JOIN churn_labels l ON l.customer_id = c.customer_id;
