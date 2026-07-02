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

    @property
    def feature_store_table(self) -> str:
        return f"{self.catalog}.{self.schema}.feature_store_cohort"

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

    # ---- Inference outputs ------------------------------------------------------
    @property
    def inference_predictions(self) -> str:
        return f"{self.catalog}.{self.schema}.inference_predictions"

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

    # ---- Thresholds -------------------------------------------------------------
    PSI_WARN_THRESHOLD: float = 0.1
    PSI_ALERT_THRESHOLD: float = 0.25
    MAPE_PROMOTION_THRESHOLD: float = 15.0   # challenger must beat champion by this
    NULL_RATE_FAIL_THRESHOLD: float = 0.05
    ROW_COUNT_DROP_FAIL_THRESHOLD: float = 0.5  # fail if >50% fewer rows than expected
    RETRAIN_NULL_RATE_THRESHOLD: float = 0.1
    RETRAIN_MAPE_THRESHOLD: float = 20.0


def get_config(catalog: str, schema: str) -> Config:
    return Config(catalog=catalog, schema=schema)
