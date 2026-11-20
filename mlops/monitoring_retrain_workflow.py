import json
import os
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from data_generator import FEATURE_COLUMNS, generate_cold_chain_dataset
from mlops.train import (
    COMPRESSOR_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    REFERENCE_DATA_PATH,
    SPOILAGE_MODEL_NAME,
    setup_mlflow,
    train_and_register,
)

REPORTS_DIR = "reports"
ARTIFACTS_DIR = "artifacts"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def task_a_prepare_new_data() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TASK A: PREPARE & VALIDATE NEW INCOMING DATA BATCH")
    print("=" * 65)

    # Generate new labelled incoming telemetry batch (heatwave shift)
    new_batch = generate_cold_chain_dataset(n_samples=600, seed=2026, heatwave_regime=True)
    new_batch.insert(0, "record_id", [f"REC-{10000 + i}" for i in range(len(new_batch))])

    required_cols = FEATURE_COLUMNS + ["time_to_spoilage", "compressor_failure"]
    missing_cols = [c for c in required_cols if c not in new_batch.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"

    missing_values = new_batch[required_cols].isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    duplicate_records = int(new_batch.duplicated().sum())
    duplicate_ids = int(new_batch["record_id"].duplicated().sum())

    target_dist = new_batch["compressor_failure"].value_counts().to_dict()
    target_pct = (new_batch["compressor_failure"].value_counts(normalize=True) * 100).round(2).to_dict()

    print(f"Dataset Size          : {len(new_batch)} rows x {len(new_batch.columns)} columns")
    print(f"Required Columns Valid: YES ({len(required_cols)} columns checked)")
    print(f"Total Missing Values  : {total_missing}")
    print(f"Duplicate Records     : {duplicate_records} | Duplicate IDs: {duplicate_ids}")
    print(f"Target Distribution   : Class 0 (Healthy)={target_dist.get(0, 0)} ({target_pct.get(0, 0)}%), "
          f"Class 1 (Failure)={target_dist.get(1, 0)} ({target_pct.get(1, 0)}%)")

    return new_batch


def evaluate_models(spoilage_model, comp_model, df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y_sp = df["time_to_spoilage"]
    y_cp = df["compressor_failure"]

    sp_preds = spoilage_model.predict(X)
    cp_preds = comp_model.predict(X)

    return {
        "regression_time_to_spoilage": {
            "mae": round(float(mean_absolute_error(y_sp, sp_preds)), 4),
            "rmse": round(float(root_mean_squared_error(y_sp, sp_preds)), 4),
            "r2": round(float(r2_score(y_sp, sp_preds)), 4),
        },
        "classification_compressor_failure": {
            "accuracy": round(float(accuracy_score(y_cp, cp_preds)), 4),
            "precision": round(float(precision_score(y_cp, cp_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_cp, cp_preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_cp, cp_preds, zero_division=0)), 4),
        },
    }








if __name__ == "__main__":
    batch_df = task_a_prepare_new_data()
    champ_m, _ = task_b_evaluate_existing_model(batch_df)
    cand_m, r_id, art_paths = task_c_retrain_candidate(batch_df)
    task_d_quality_gate(champ_m, cand_m, r_id, art_paths)
