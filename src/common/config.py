"""
src/common/config.py

Centralised configuration so every task reads/writes the same Unity Catalog
paths. Keeping this in one module (instead of hardcoding table names in every
notebook, like the original single-notebook version did with raw s3:// paths)
is what makes it safe to point dev/staging/prod at different catalogs purely
via job parameters, with zero code changes.
"""
from dataclasses import dataclass


@dataclass
class Config:
    catalog: str
    schema: str
    secret_scope: str = "scf-cohort-notifications"  # overridden per env via job widget
    feature_schema: str = "feature_store"            # separate schema for all feature tables
    model_schema: str   = "ml_models"                # separate schema for model registry artifacts

    # ---- Feature Store schema tables -----------------------------------------
    @property
    def feature_store_cohort(self) -> str:
        """Existing 3 cohort features (int_rate, best_buy, delta_to_best_buy etc.)"""
        return f"{self.catalog}.{self.feature_schema}.feature_store_cohort"

    # ---- Bronze / source tables (replace the original s3:// parquet reads) --
    @property
    def raw_base_data(self) -> str:
        return f"{self.catalog}.{self.schema}.raw_base_data"

    @property
    def raw_best_buy(self) -> str:
        return f"{self.catalog}.{self.schema}.raw_moneyfacts_best_buy"

    @property
    def raw_qrm_products(self) -> str:
        return f"{self.catalog}.{self.schema}.raw_qrm_products"

    # ---- Silver (post data-preprocessing) ------------------------------------
    @property
    def silver_agg_cohort(self) -> str:
        return f"{self.catalog}.{self.schema}.silver_agg_cohort"

    # ---- GLM flow-ratio lookup tables (built by 00_feature_engineering.py) ---
    # Receipts & outflows are indexed by cohort age × rate bucket; transfers
    # only by rate bucket (transfer share depends on delta_to_best_buy alone).
    @property
    def receipts_lookup(self) -> str:
        return f"{self.catalog}.{self.feature_schema}.receipts_ratio_lookup"

    @property
    def outflows_lookup(self) -> str:
        return f"{self.catalog}.{self.feature_schema}.outflows_ratio_lookup"

    @property
    def transfers_lookup(self) -> str:
        return f"{self.catalog}.{self.feature_schema}.transfers_ratio_lookup"

    @property
    def feature_store_table(self) -> str:
        return f"{self.catalog}.{self.feature_schema}.feature_store_cohort"

    # ---- Training artifacts ---------------------------------------------------
    @property
    def training_dataset(self) -> str:
        return f"{self.catalog}.{self.schema}.training_dataset"

    @property
    def eval_metrics(self) -> str:
        return f"{self.catalog}.{self.schema}.model_eval_metrics"

    @property
    def dataset_versions(self) -> str:
        return f"{self.catalog}.{self.schema}.dataset_versions"

    # ---- Audit / governance tables (required for regulatory audit) -----------
    @property
    def model_approvals(self) -> str:
        """FMC approval/rejection decision, approver name, model version, timestamp."""
        return f"{self.catalog}.{self.schema}.model_approvals"

    @property
    def model_cards(self) -> str:
        """Auto-generated card: model name, version, run ID, features, labels, portfolio MAPE."""
        return f"{self.catalog}.{self.schema}.model_cards"

    @property
    def model_coefficients(self) -> str:
        """Per-product GLM coefficient summary (term, estimate, SE, p-value, CI, GOF) for model-risk sign-off."""
        return f"{self.catalog}.{self.schema}.model_coefficients"

    @property
    def deployment_history(self) -> str:
        """Every endpoint create/update: version deployed, rollout %, timestamp."""
        return f"{self.catalog}.{self.schema}.deployment_history"

    @property
    def rollback_events(self) -> str:
        """Rollback reason, version reverted to, smoke test result."""
        return f"{self.catalog}.{self.schema}.rollback_events"

    @property
    def training_feature_snapshots(self) -> str:
        """Full feature snapshot frozen at training time — survives Delta VACUUM."""
        return f"{self.catalog}.{self.schema}.training_feature_snapshots"

    # ---- Inference outputs ------------------------------------------------------
    @property
    def inference_predictions(self) -> str:
        return f"{self.catalog}.{self.schema}.inference_predictions"

    @property
    def inference_predictions_product_month(self) -> str:
        """Per-cohort forecasts summed up to product x reporting_period."""
        return f"{self.catalog}.{self.schema}.inference_predictions_product_month"

    @property
    def inference_features(self) -> str:
        return f"{self.catalog}.{self.schema}.inference_features"

    # ---- Monitoring / drift / quality ------------------------------------------
    @property
    def monitoring_metrics(self) -> str:
        return f"{self.catalog}.{self.schema}.monitoring_metrics"

    @property
    def drift_results(self) -> str:
        return f"{self.catalog}.{self.schema}.drift_results"

    @property
    def data_quality_results(self) -> str:
        return f"{self.catalog}.{self.schema}.data_quality_results"

    @property
    def volume_reference_data(self) -> str:
        # Reference (training-time) feature snapshot used as the drift baseline.
        return f"{self.catalog}.{self.schema}.reference_feature_snapshot"

    @property
    def prediction_reference_data(self) -> str:
        # Reference (baseline) prediction distribution used for prediction /
        # output drift. Frozen from the first scored batch, refreshed
        # deliberately after a validated retrain (never auto-chases the data).
        return f"{self.catalog}.{self.schema}.reference_prediction_snapshot"

    # ---- Thresholds -------------------------------------------------------------
    PSI_WARN_THRESHOLD: float = 0.1
    PSI_ALERT_THRESHOLD: float = 0.25
    MAPE_PROMOTION_THRESHOLD: float = 15.0   # challenger must beat champion by this
    NULL_RATE_FAIL_THRESHOLD: float = 0.05
    ROW_COUNT_DROP_FAIL_THRESHOLD: float = 0.5  # fail if >50% fewer rows than expected
    RETRAIN_NULL_RATE_THRESHOLD: float = 0.1
    RETRAIN_MAPE_THRESHOLD: float = 20.0
    # GLM extrapolation guard: GLMs extrapolate linearly, so predictions made
    # where delta_to_best_buy falls outside the training support are unreliable.
    # Alert when more than this fraction of the batch is out-of-range.
    EXTRAPOLATION_ALERT_RATE: float = 0.05


def get_config(
    catalog: str,
    schema: str,
    secret_scope: str   = "scf-cohort-notifications",
    feature_schema: str = "feature_store",
    model_schema: str   = "ml_models",
) -> Config:
    return Config(
        catalog=catalog,
        schema=schema,
        secret_scope=secret_scope,
        feature_schema=feature_schema,
        model_schema=model_schema,
    )
