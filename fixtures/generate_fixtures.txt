"""
generate_fixtures.py

Creates synthetic SCF cohort data that matches the schema the pipeline expects,
so the bundle can be run end-to-end without access to the real S3 PII data.

Produces 4 CSVs under fixtures/savingscashflow/:
  - base_data_training.csv     : account-level history, cohorts 2023-06..2024-12,
                                  reporting periods 2023-06..2025-06 (used by the
                                  training pipeline; matches cutoff_period=2025-01)
  - base_data_inference.csv    : the "latest batch" - reporting periods
                                  2025-07..2025-09 for the same cohorts, appended
                                  monthly in real life, used by the inference pipeline
  - moneyfacts_best_buy.csv    : market best-buy rate by (period, mapping_category),
                                  covering the full period range above
  - qrm_products.csv           : product code -> QRM Category mapping

Column names intentionally match the original notebook's raw extract:
  'account id', 'product code', 'opening date', 'reporting period',
  'period balance', '£ withdrawal', '£ receipt', '£ internal transfer',
  'interest rate', 'account tenure (months)'

Re-runnable and deterministic (fixed seed) so CI can regenerate if needed.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

OUT_DIR = "savingscashflow"

# ---------------------------------------------------------------------------
# Product catalogue - 5 products spanning the QRM categories the notebook's
# best_buy_mapping dict knows how to translate.
# ---------------------------------------------------------------------------
PRODUCTS = [
    # product_code, QRM Category,           best_buy mapping_category, base_rate
    ("P001", "Variable ISA (Online)",     "EA ISA (Online)",         0.032),
    ("P002", "Variable NonISA (Online)",  "EA Non-ISA (Online)",     0.028),
    ("P003", "HL ISA",                    "EA ISA (Online)",         0.030),
    ("P004", "Restricted ISA",            "Restricted ISA",          0.025),
    ("P005", "Restricted NonISA",         "Restricted Non-ISA",      0.022),
]

qrm_df = pd.DataFrame(
    [{"Product Code": p[0], "QRM Category": p[1]} for p in PRODUCTS]
)
qrm_df.to_csv(f"{OUT_DIR}/qrm_products.csv", index=False)

# ---------------------------------------------------------------------------
# Best-buy market rate table: monthly, per mapping_category, Jun-2023..Sep-2025
# ---------------------------------------------------------------------------
all_periods = pd.period_range("2023-06", "2025-09", freq="M")
mapping_categories = sorted(set(p[2] for p in PRODUCTS))

bb_rows = []
for cat in mapping_categories:
    drift = rng.normal(0, 0.0015)
    rate = rng.uniform(0.03, 0.045)
    for period in all_periods:
        rate = max(0.005, rate + drift + rng.normal(0, 0.0008))
        bb_rows.append({"period": str(period), "mapping_category": cat, "best_buy": round(rate, 5)})

best_buy_df = pd.DataFrame(bb_rows)
best_buy_df.to_csv(f"{OUT_DIR}/moneyfacts_best_buy.csv", index=False)


# ---------------------------------------------------------------------------
# Account-level base data. For each product, spin up cohorts (accounts opened
# in a given month) and simulate monthly balance/withdrawal/receipt/transfer
# behaviour with attrition, so downstream cohort aggregation produces
# realistic-looking decay curves for the GLMs to fit.
# ---------------------------------------------------------------------------
def simulate_product(product_code, qrm_category, base_rate, cohort_months, n_accounts_per_cohort):
    rows = []
    account_seq = 0
    for cohort_month in cohort_months:
        cohort_period = pd.Period(cohort_month, freq="M")
        n_accounts = int(n_accounts_per_cohort * rng.uniform(0.8, 1.2))
        opening_balances = rng.lognormal(mean=8.5, sigma=0.6, size=n_accounts)  # ~£3k-£15k typical

        for acct_i in range(n_accounts):
            account_seq += 1
            account_id = f"{product_code}-{cohort_month}-{acct_i:04d}"
            opening_date = f"{cohort_month}-{rng.integers(1, 28):02d}"
            balance = opening_balances[acct_i]

            # attrition: each account has a monthly chance of closing, rising
            # slowly with tenure (matches typical savings cohort decay)
            max_months = int((pd.Period("2025-09", freq="M") - cohort_period).n + 1)
            for m in range(1, max_months + 1):
                reporting_period = cohort_period + (m - 1)
                if reporting_period > pd.Period("2025-09", freq="M"):
                    break

                close_prob = min(0.03 + 0.002 * m, 0.15)
                if balance <= 1 or (m > 1 and rng.random() < close_prob):
                    break  # account closed / drained - stop emitting rows

                interest_rate = max(0.001, base_rate + rng.normal(0, 0.003))
                receipt = max(0, rng.normal(balance * 0.04, balance * 0.02))
                withdrawal = max(0, rng.normal(balance * 0.06, balance * 0.03))
                transfer = max(0, rng.normal(balance * 0.01, balance * 0.01))

                new_balance = max(0, balance + receipt - withdrawal - transfer)

                rows.append({
                    "account id": account_id,
                    "product code": product_code,
                    "opening date": opening_date,
                    "reporting period": str(reporting_period),
                    "period balance": round(new_balance, 2),
                    "£ withdrawal": round(withdrawal, 2),
                    "£ receipt": round(receipt, 2),
                    "£ internal transfer": round(transfer, 2),
                    "interest rate": round(interest_rate, 5),
                    "account tenure (months)": m,
                })
                balance = new_balance
    return pd.DataFrame(rows)


TRAIN_COHORT_MONTHS = [f"2023-{m:02d}" if m <= 12 else f"2024-{m-12:02d}" for m in range(6, 19)]  # 2023-06..2024-06
ACCOUNTS_PER_COHORT = 40

all_data = []
for product_code, qrm_category, mapping_category, base_rate in PRODUCTS:
    df = simulate_product(product_code, qrm_category, base_rate, TRAIN_COHORT_MONTHS, ACCOUNTS_PER_COHORT)
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
full_df["reporting period"] = full_df["reporting period"]

# Training extract: everything up to (and including) reporting period 2025-06
train_mask = full_df["reporting period"] <= "2025-06"
base_data_training = full_df[train_mask].copy()
base_data_training.to_csv(f"{OUT_DIR}/base_data_training.csv", index=False)

# Inference extract: the "next" 3 months of scoring data (2025-07..2025-09) -
# simulates the incremental monthly batch the inference pipeline would land
base_data_inference = full_df[~train_mask].copy()
base_data_inference.to_csv(f"{OUT_DIR}/base_data_inference.csv", index=False)

print(f"qrm_products.csv:        {len(qrm_df)} rows")
print(f"moneyfacts_best_buy.csv: {len(best_buy_df)} rows")
print(f"base_data_training.csv:  {len(base_data_training)} rows, "
      f"{base_data_training['account id'].nunique()} accounts")
print(f"base_data_inference.csv: {len(base_data_inference)} rows, "
      f"{base_data_inference['account id'].nunique()} accounts")
