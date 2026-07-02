"""
src/common/pyfunc_model.py

The GLMs themselves aren't what inference calls at serving time - the
*prediction grid / composite pct tables* built once per product during
training are (that's exactly how the original notebook's projection step
worked: `rec_pct.loc[month, rate_bin]`). So the pyfunc model packages the
three pct tables per product and replays the same lookup + flow-projection
logic (`BuildProjections`) inside `predict()`. This keeps train-time and
inference-time projection math as one shared code path
(src/common/glm_core.py) instead of two.

Input schema expected at predict() time (one row per cohort x period to
project):
  ['product', 'cohort', 'reporting_period', 'months_since_start',
   'dbb_range', 'balance_lag_1']
"""
import mlflow
import pandas as pd

from src.common.glm_core import BuildProjections


class SCFCohortModel(mlflow.pyfunc.PythonModel):
    def __init__(self, product_tables: dict, trained_at: str, cutoff_period: str):
        """
        product_tables: {product_name: {'rec_pct_tbl': df, 'outflow_pct_tbl': df, 'transfer_pct_tbl': df}}
        """
        self.product_tables = product_tables
        self.trained_at = trained_at
        self.cutoff_period = cutoff_period

    def predict(self, context, model_input: pd.DataFrame):
        projector = BuildProjections()
        outputs = []
        for product, tables in self.product_tables.items():
            subset = model_input[model_input['product'] == product]
            if subset.empty:
                continue
            proj = projector.project_balances_for_all_cohorts(
                act_df=subset,
                rec_pct=tables['rec_pct_tbl'],
                outflow_pct=tables['outflow_pct_tbl'],
                transfer_pct=tables['transfer_pct_tbl'],
                start_period=subset['reporting_period'].min(),
                end_period=subset['reporting_period'].max(),
                product=product,
            )
            outputs.append(proj)
        if not outputs:
            return pd.DataFrame()
        return pd.concat(outputs, ignore_index=True)
