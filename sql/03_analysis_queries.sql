-- SYNTHETIC DATA. Business-facing analysis queries (run through src/build_sqlite.py).

-- Q1: churn rate and monthly revenue lost by contract type
SELECT c.contract_type, COUNT(*) AS customers,
       ROUND(100.0 * AVG(l.churned_30d), 2)                              AS churn_rate_pct,
       ROUND(SUM(CASE WHEN l.churned_30d = 1 THEN c.monthly_charge END), 0) AS monthly_revenue_lost
FROM customers c JOIN churn_labels l USING (customer_id)
GROUP BY c.contract_type ORDER BY churn_rate_pct DESC;

-- Q2: churn rate by tenure bucket (window-free cohort view)
SELECT CASE WHEN tenure_months <= 6 THEN '0-6' WHEN tenure_months <= 12 THEN '7-12'
            WHEN tenure_months <= 24 THEN '13-24' WHEN tenure_months <= 48 THEN '25-48' ELSE '49+' END AS tenure_bucket,
       COUNT(*) AS customers, ROUND(100.0 * AVG(l.churned_30d), 2) AS churn_rate_pct
FROM customers c JOIN churn_labels l USING (customer_id)
GROUP BY tenure_bucket ORDER BY MIN(tenure_months);

-- Q3: ticket-burst customers (>=2 tickets in last 30 days) - volume, churn and revenue exposed
WITH recent AS (
  SELECT customer_id, COUNT(*) AS n30 FROM support_tickets
  WHERE julianday('2025-12-31') - julianday(created_date) <= 30 GROUP BY customer_id)
SELECT CASE WHEN COALESCE(r.n30,0) >= 2 THEN '2+ tickets' WHEN COALESCE(r.n30,0) = 1 THEN '1 ticket' ELSE '0 tickets' END AS segment,
       COUNT(*) AS customers, ROUND(100.0 * AVG(l.churned_30d), 2) AS churn_rate_pct,
       ROUND(SUM(c.monthly_charge), 0) AS monthly_revenue_exposed
FROM customers c JOIN churn_labels l USING (customer_id) LEFT JOIN recent r USING (customer_id)
GROUP BY segment ORDER BY churn_rate_pct DESC;

-- Q4: window function - rank regions by churn within each contract type
SELECT * FROM (
  SELECT c.contract_type, c.region, ROUND(100.0 * AVG(l.churned_30d), 2) AS churn_rate_pct,
         RANK() OVER (PARTITION BY c.contract_type ORDER BY AVG(l.churned_30d) DESC) AS rank_in_contract
  FROM customers c JOIN churn_labels l USING (customer_id)
  GROUP BY c.contract_type, c.region)
WHERE rank_in_contract <= 2 ORDER BY contract_type, rank_in_contract;
