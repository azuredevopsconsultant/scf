"""
src/common/glm_core.py

Direct port of the modelling classes from the original `scf_cohort_matrix.ipynb`
(DataPrep, GLMTrainer, PredictionBuilder, BuildProjections, Evaluator), with
RATE_BINS/RATE_LABELS hoisted to module level so training and inference share
one definition instead of two copy-pasted ones (a real risk in the original
notebook, where the bins were re-declared in cells 37 and 46).

Nothing here is Databricks-specific - it's plain pandas/statsmodels, so it's
identical to what was validated in the source notebook. All orchestration
(reading/writing UC tables, MLflow, widgets) stays in the task scripts.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import PatsyError

RATE_BINS = [-np.inf, -1, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0, 0.05, np.inf]
RATE_LABELS = [
    '(-inf, -1)', '[-1, -0.75)', '[-0.75, -0.5)', '[-0.5, -0.4)',
    '[-0.4, -0.3)', '[-0.3, -0.2)', '[-0.2, -0.1)', '[-0.1, -0.05)',
    '[-0.05, 0)', '[0, 0.05)', '[0.05, inf)',
]


class DataPrep:
    """Train/test split, month-2 drop, IQR outlier removal - per product."""

    def __init__(self, df, product, cutoff_period='2025-01', drop_month2=True):
        self.df = df.copy()
        self.product = product
        self.cutoff_period = cutoff_period
        self.drop_month2 = drop_month2
        self.train = None
        self.test = None

    def remove_outliers(self, df, column):
        q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return df.loc[~((df[column] < lower) | (df[column] > upper))].copy()

    def split_train_test(self):
        df = self.df[self.df['product'] == self.product].copy()
        cutoff = pd.Period(self.cutoff_period, freq='M')
        self.train = df[df['reporting_period'] < cutoff].copy()
        self.test = df[df['reporting_period'] >= cutoff].copy()
        if self.drop_month2:
            self.train = self.train[self.train['months_since_start'] != 2]
        return self.train, self.test

    def build_training_set(self):
        if self.train is None:
            self.split_train_test()
        train = self.train.copy()
        train_rec = self.remove_outliers(train, 'rec_prop')
        train_outflow = self.remove_outliers(train, 'outflow_prop')
        train_wd = self.remove_outliers(train, 'withdrawal_prop_of_outflow')
        return train, train_rec, train_outflow, train_wd


class GLMTrainer:
    """Fits the three per-product GLMs: outflow, withdrawal-prop, receipts."""

    def __init__(self, train_rec, train_outflow, train_wd):
        self.train_rec = train_rec.copy()
        self.train_outflow = train_outflow.copy()
        self.train_wd = train_wd.copy()
        self.outflow_results = None
        self.wd_results = None
        self.rec_results = None

    def fit_all(self):
        self.outflow_results = smf.glm(
            formula='outflow_prop ~ cr(months_since_start, df=4) + delta_to_best_buy',
            data=self.train_outflow,
            family=sm.families.Binomial(),
        ).fit(cov_type='HC0')

        self.wd_results = smf.glm(
            formula='withdrawal_prop_of_outflow ~ delta_to_best_buy',
            data=self.train_wd,
            family=sm.families.Binomial(),
        ).fit(cov_type='HC0')

        if (self.train_rec['rec_prop'] <= 0).any():
            rec_family = sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
        else:
            rec_family = sm.families.Gamma(link=sm.families.links.Log())

        self.rec_results = smf.glm(
            formula='rec_prop ~ cr(months_since_start, df=4) + delta_to_best_buy',
            data=self.train_rec,
            family=rec_family,
        ).fit(cov_type='HC0')

        return self.rec_results, self.outflow_results, self.wd_results


class PredictionBuilder:
    def representative_delta_from_train(self, train):
        midpoints = []
        for lo, hi in zip(RATE_BINS[:-1], RATE_BINS[1:]):
            if lo == -np.inf:
                midpoints.append(hi - 0.5)
            elif hi == np.inf:
                midpoints.append(lo + 0.025)
            else:
                midpoints.append((lo + hi) / 2)
        mid_series = pd.Series(midpoints, index=RATE_LABELS)
        means = train.groupby('dbb_range', observed=True)['delta_to_best_buy'].median()
        return means.fillna(mid_series)

    def build_prediction_grid(self, train, month_start=2, month_end=63):
        reps = self.representative_delta_from_train(train)
        months = np.arange(month_start, month_end + 1)
        grid = pd.MultiIndex.from_product(
            [months, RATE_LABELS], names=['months_since_start', 'rate_bin']
        ).to_frame(index=False)
        rep_map = dict(zip(RATE_LABELS, reps.values))
        grid['delta_to_best_buy'] = grid['rate_bin'].map(rep_map)
        return grid

    def _training_bounds_months_since_start(self, res, col='months_since_start'):
        tr = res.model.data.frame[col]
        return float(np.nanmin(tr)), float(np.nanmax(tr))

    def _predict_all(self, rec_results, outflow_results, wd_results, df):
        def clamp_pred(pred, family):
            pred = np.asarray(pred, float)
            if isinstance(family, sm.families.Binomial):
                return np.clip(pred, 0, 1)
            return np.maximum(pred, 0)

        out = df.copy()
        out['rec_pred'] = clamp_pred(rec_results.predict(out), rec_results.model.family)
        out['outflow_pred'] = clamp_pred(outflow_results.predict(out), outflow_results.model.family)
        out['wd_pred'] = clamp_pred(wd_results.predict(out), wd_results.model.family)
        return out

    def glm_predict_grid(self, rec_results, outflow_results, wd_results, grid, xcol='months_since_start'):
        df = grid.copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df[xcol] = pd.to_numeric(df[xcol], errors='coerce')

        try:
            return self._predict_all(rec_results, outflow_results, wd_results, df)
        except (PatsyError, NotImplementedError) as e:
            lo_rec, hi_rec = self._training_bounds_months_since_start(rec_results, xcol)
            lo_out, hi_out = self._training_bounds_months_since_start(outflow_results, xcol)
            lo_wd, hi_wd = self._training_bounds_months_since_start(wd_results, xcol)
            lo, hi = max(lo_rec, lo_out, lo_wd), min(hi_rec, hi_out, hi_wd)
            if lo > hi:
                raise ValueError(f'No overlapping {xcol} range across models') from e
            clipped = df.copy()
            clipped[xcol] = clipped[xcol].clip(lo, hi)
            try:
                return self._predict_all(rec_results, outflow_results, wd_results, clipped)
            except Exception as e2:
                raise RuntimeError(f'Prediction failed after clipping {xcol} to [{lo}, {hi}].') from e2

    def compute_composites(self, df):
        out = df.copy()
        out['rec_pct'] = out['rec_pred'] * 100
        out['outflow_pct'] = (out['outflow_pred'] * (1 + out['rec_pred'])) * 100
        out['transfer_pct'] = (1 - out['wd_pred']) * 100
        return out

    def pivot_ratio_table(self, df, ratio):
        df = df.drop_duplicates()
        tbl = df.pivot(index='months_since_start', columns='rate_bin', values=ratio)
        return tbl.reindex(columns=RATE_LABELS)


class BuildProjections:
    """Turns predicted ratio tables into projected receipts/withdrawals/transfers/balance."""

    def clamp_month(self, m, pred_df):
        return max(pred_df.index.min(), min(m, pred_df.index.max()))

    def project_balances(self, act_df, rec_pct, outflow_pct, transfer_pct, start_period, end_period, product):
        df = act_df[
            (act_df['product'] == product)
            & (act_df['reporting_period'] >= start_period)
            & (act_df['reporting_period'] <= end_period)
        ].copy()
        df = df.sort_values('reporting_period').reset_index(drop=True)
        df['balance_lag_1'] = pd.to_numeric(df['balance_lag_1'], errors='coerce')
        for col in ['receipts_pred', 'withdrawals_pred', 'transfers_pred', 'balance_pred']:
            df[col] = np.nan

        for i, row in df.iterrows():
            m, d = row['months_since_start'], row['dbb_range']
            bal_lag = df.at[i, 'balance_lag_1']

            if m == 1:
                df.at[i, 'receipts_pred'] = 0
                df.at[i, 'withdrawals_pred'] = 0
                df.at[i, 'transfers_pred'] = 0
                df.at[i, 'balance_pred'] = bal_lag
                continue
            if pd.isna(bal_lag):
                df.at[i, 'balance_pred'] = np.nan
                continue

            m_clamped = self.clamp_month(m, rec_pct)
            rct = rec_pct.loc[m_clamped, d] / 100
            otf = outflow_pct.loc[m_clamped, d] / 100
            trf = transfer_pct.loc[m_clamped, d] / 100

            receipts = bal_lag * rct
            outflows = bal_lag * otf
            transfers = outflows * trf
            withdrawals = outflows - transfers
            new_balance = bal_lag + receipts - withdrawals - transfers

            df.at[i, 'receipts_pred'] = receipts
            df.at[i, 'withdrawals_pred'] = withdrawals
            df.at[i, 'transfers_pred'] = transfers
            df.at[i, 'balance_pred'] = new_balance
            df.at[i, 'rec_pct_pred'] = rct
            df.at[i, 'outflow_pct_pred'] = otf
            df.at[i, 'transfer_pct_pred'] = trf

            if i + 1 < len(df):
                df.at[i + 1, 'balance_lag_1'] = new_balance

        return df

    def project_balances_for_all_cohorts(self, act_df, rec_pct, outflow_pct, transfer_pct, start_period, end_period, product):
        cohorts = act_df[act_df['product'] == product]['cohort'].unique()
        all_results = []
        for cohort in cohorts:
            cohort_df = act_df[(act_df['product'] == product) & (act_df['cohort'] == cohort)]
            proj = self.project_balances(cohort_df, rec_pct, outflow_pct, transfer_pct, start_period, end_period, product)
            all_results.append(proj)
        return pd.concat(all_results, ignore_index=True)


class Evaluator:
    def mape(self, df, actual_col, pred_col):
        data = df[[actual_col, pred_col]].dropna()
        # Guard the zero denominator: |actual| == 0 would make APE inf and
        # poison the mean (a real risk for withdrawals/transfers that can be 0
        # in a month). Drop those rows; return NaN if nothing scoreable remains.
        data = data[data[actual_col] != 0]
        if data.empty:
            return float("nan")
        return (data[actual_col] - data[pred_col]).abs().div(data[actual_col].abs()).mean() * 100

    def rmse(self, df, actual_col, pred_col):
        data = df[[actual_col, pred_col]].dropna()
        return np.sqrt(((data[actual_col] - data[pred_col]) ** 2).mean())


class ModelingPipeline:
    """End-to-end per-product workflow: prep -> fit -> grid -> composites -> (optional) projection+eval."""

    def __init__(self, df):
        self.df = df.copy()

    def run_for_product(
        self, PRODUCT, cutoff_period='2025-01', drop_month2=True, month_start=2, month_end=50,
        proj_start_period=None, proj_end_period=None, proj_cohort_cutoff='2024-12', save_csv=False,
    ):
        prep = DataPrep(self.df, PRODUCT, cutoff_period, drop_month2)
        train, train_rec, train_outflow, train_wd = prep.build_training_set()

        glm = GLMTrainer(train_rec, train_outflow, train_wd)
        rec_results, outflow_results, wd_results = glm.fit_all()

        stage = PredictionBuilder()
        grid = stage.build_prediction_grid(train, month_start=month_start, month_end=month_end)
        pred_grid = stage.glm_predict_grid(rec_results, outflow_results, wd_results, grid)
        comp_grid = stage.compute_composites(pred_grid)

        rec_pct_tbl = stage.pivot_ratio_table(comp_grid, 'rec_pct')
        outflow_pct_tbl = stage.pivot_ratio_table(comp_grid, 'outflow_pct')
        transfer_pct_tbl = stage.pivot_ratio_table(comp_grid, 'transfer_pct')

        results = {
            'product': PRODUCT,
            'train': train, 'train_rec': train_rec, 'train_outflow': train_outflow, 'train_wd': train_wd,
            'rec_results': rec_results, 'outflow_results': outflow_results, 'wd_results': wd_results,
            'grid': grid, 'pred_grid': pred_grid, 'comp_grid': comp_grid,
            'rec_pct_tbl': rec_pct_tbl, 'outflow_pct_tbl': outflow_pct_tbl, 'transfer_pct_tbl': transfer_pct_tbl,
        }

        if proj_start_period and proj_end_period:
            projector = BuildProjections()
            act_df = self.df.copy()
            if 'dbb_range' not in act_df.columns:
                act_df['dbb_range'] = pd.cut(act_df['delta_to_best_buy'], bins=RATE_BINS, labels=RATE_LABELS)

            projected_df = projector.project_balances_for_all_cohorts(
                act_df, rec_pct_tbl, outflow_pct_tbl, transfer_pct_tbl, proj_start_period, proj_end_period, PRODUCT
            )
            projected_df = projected_df[~(projected_df['cohort'] > proj_cohort_cutoff)]

            projected_flows = projected_df.groupby('reporting_period')[
                ['balance', 'balance_pred', 'receipts', 'receipts_pred',
                 'withdrawals', 'withdrawals_pred', 'transfers', 'transfers_pred']
            ].sum()

            results['projected_df'] = projected_df
            results['projected_flows'] = projected_flows

            evaluator = Evaluator()
            results['metrics'] = {
                'mape_balance': evaluator.mape(projected_flows, 'balance', 'balance_pred'),
                'mape_receipts': evaluator.mape(projected_flows, 'receipts', 'receipts_pred'),
                'mape_withdrawals': evaluator.mape(projected_flows, 'withdrawals', 'withdrawals_pred'),
                'mape_transfers': evaluator.mape(projected_flows, 'transfers', 'transfers_pred'),
                'rmse_balance': evaluator.rmse(projected_flows, 'balance', 'balance_pred'),
                'rmse_receipts': evaluator.rmse(projected_flows, 'receipts', 'receipts_pred'),
                'rmse_withdrawals': evaluator.rmse(projected_flows, 'withdrawals', 'withdrawals_pred'),
                'rmse_transfers': evaluator.rmse(projected_flows, 'transfers', 'transfers_pred'),
            }

            if save_csv:
                os.makedirs('Results', exist_ok=True)
                rec_pct_tbl.to_csv(f'Results/{PRODUCT}_rec_pct_tbl.csv')
                outflow_pct_tbl.to_csv(f'Results/{PRODUCT}_outflow_pct_tbl.csv')
                transfer_pct_tbl.to_csv(f'Results/{PRODUCT}_transfer_pct_tbl.csv')
                projected_flows.to_csv(f'Results/{PRODUCT}_projected_flows.csv')

        return results
