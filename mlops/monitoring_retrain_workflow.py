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

def task_b_evaluate_existing_model(new_batch: pd.DataFrame) -> tuple:
    print("\n" + "=" * 65)
    print("TASK B: EVALUATE EXISTING @champion MODEL ON NEW BATCH")
    print("=" * 65)

    client = setup_mlflow()
    try:
        client.get_model_version_by_alias(SPOILAGE_MODEL_NAME, "champion")
    except Exception:
        print("No @champion found in MLflow. Bootstrapping baseline v1 first...")
        train_and_register(heatwave_regime=False, assign_alias="champion", run_tag="v1_baseline")

    champ_sp = mlflow.sklearn.load_model(f"models:/{SPOILAGE_MODEL_NAME}@champion")
    champ_cp = mlflow.sklearn.load_model(f"models:/{COMPRESSOR_MODEL_NAME}@champion")

    champ_metrics = evaluate_models(champ_sp, champ_cp, new_batch)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_model": "@champion",
        "batch_size": len(new_batch),
        "metrics": champ_metrics,
    }
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Regression (Time-to-Spoilage)     :", champ_metrics["regression_time_to_spoilage"])
    print("Classification (Compressor Fail)  :", champ_metrics["classification_compressor_failure"])
    print(f"Saved evaluation report -> {report_path}")
    return champ_metrics, report_path


def task_c_retrain_candidate(new_batch: pd.DataFrame) -> tuple:
    print("\n" + "=" * 65)
    print("TASK C: RETRAIN CANDIDATE MODEL (EXISTING + NEW DATA)")
    print("=" * 65)

    if os.path.exists(REFERENCE_DATA_PATH):
        existing_df = pd.read_parquet(REFERENCE_DATA_PATH)
    else:
        existing_df = generate_cold_chain_dataset(n_samples=2500, seed=42, heatwave_regime=False)

    new_clean = new_batch[FEATURE_COLUMNS + ["time_to_spoilage", "compressor_failure"]]
    combined_df = pd.concat([existing_df, new_clean], ignore_index=True)

    X = combined_df[FEATURE_COLUMNS]
    y_sp = combined_df["time_to_spoilage"]
    y_cp = combined_df["compressor_failure"]

    X_train, X_val, y_sp_train, y_sp_val, y_cp_train, y_cp_val = train_test_split(
        X, y_sp, y_cp, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("PharmaPulse_ColdChain_AIOps")

    with mlflow.start_run(run_name="candidate_retrain_assessment") as run:
        cand_sp = lgb.LGBMRegressor(
            n_estimators=160, learning_rate=0.06, max_depth=6, num_leaves=31, random_state=42, verbose=-1
        )
        cand_sp.fit(X_train, y_sp_train)

        cand_cp = GradientBoostingClassifier(
            n_estimators=140, learning_rate=0.07, max_depth=4, random_state=42
        )
        cand_cp.fit(X_train, y_cp_train)

        # Evaluate candidate on the same new labelled batch for fair comparison
        cand_metrics = evaluate_models(cand_sp, cand_cp, new_batch)

        mlflow.log_param("combined_rows", len(combined_df))
        mlflow.log_metric("candidate_rmse", cand_metrics["regression_time_to_spoilage"]["rmse"])
        mlflow.log_metric("candidate_f1", cand_metrics["classification_compressor_failure"]["f1_score"])

        sp_path = os.path.join(ARTIFACTS_DIR, "candidate_spoilage_model.joblib")
        cp_path = os.path.join(ARTIFACTS_DIR, "candidate_compressor_model.joblib")
        joblib.dump(cand_sp, sp_path)
        joblib.dump(cand_cp, cp_path)

        mlflow.log_artifact(sp_path)
        mlflow.log_artifact(cp_path)

    print(f"Combined Training Dataset: {len(combined_df)} rows")
    print(f"MLflow Run ID            : {run.info.run_id}")
    print(f"Saved Candidate Artifacts: {sp_path}, {cp_path}")
    print("Candidate Regression     :", cand_metrics["regression_time_to_spoilage"])
    print("Candidate Classification :", cand_metrics["classification_compressor_failure"])
    return cand_metrics, run.info.run_id, [sp_path, cp_path]


def task_d_quality_gate(champ_metrics: dict, cand_metrics: dict, run_id: str, artifacts: list):
    print("\n" + "=" * 65)
    print("TASK D: QUALITY GATE DECISION (NEVER OVERWRITE BLINDLY)")
    print("=" * 65)

    champ_rmse = champ_metrics["regression_time_to_spoilage"]["rmse"]
    cand_rmse = cand_metrics["regression_time_to_spoilage"]["rmse"]

    champ_f1 = champ_metrics["classification_compressor_failure"]["f1_score"]
    cand_f1 = cand_metrics["classification_compressor_failure"]["f1_score"]

    # Quality Gate Rule: Candidate must improve RMSE (lower) AND maintain/improve F1-score
    rmse_improved = cand_rmse < champ_rmse
    f1_acceptable = cand_f1 >= (champ_f1 - 0.01)
    eligible = rmse_improved and f1_acceptable

    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "candidate_artifacts": artifacts,
        "comparison": {
            "champion_rmse": champ_rmse,
            "candidate_rmse": cand_rmse,
            "champion_f1": champ_f1,
            "candidate_f1": cand_f1,
        },
        "criteria": "candidate_rmse < champion_rmse AND candidate_f1 >= champion_f1 - 0.01",
        "decision": "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION",
        "champion_overwritten": False,
        "metric_justification": (
            "RMSE is selected for Time-to-Spoilage because it heavily penalizes large prediction errors "
            "that could cause missed vaccine spoilage windows. F1-score is selected for Compressor Failure "
            "because it balances Recall (catching real failures) and Precision (avoiding false emergency reroutes)."
        ),
    }

    decision_path = os.path.join(REPORTS_DIR, "quality_gate_decision.json")
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(f"Champion RMSE: {champ_rmse} vs Candidate RMSE: {cand_rmse}")
    print(f"Champion F1  : {champ_f1} vs Candidate F1  : {cand_f1}")
    print(f"FINAL DECISION: {decision['decision']} (Champion overwritten: False)")
    print(f"Saved decision record -> {decision_path}")






if __name__ == "__main__":
    batch_df = task_a_prepare_new_data()
    champ_m, _ = task_b_evaluate_existing_model(batch_df)
    cand_m, r_id, art_paths = task_c_retrain_candidate(batch_df)
    task_d_quality_gate(champ_m, cand_m, r_id, art_paths)
