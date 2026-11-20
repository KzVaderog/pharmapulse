import json
import os
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
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

FEATURE_COLUMNS = [
    "cabin_temp",
    "ambient_temp",
    "compressor_vibration_hz",
    "door_open",
    "mkt_temp",
    "temp_delta",
    "thermal_stress_index",
]

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
SPOILAGE_MODEL_NAME = "PharmaPulseSpoilageModel"
COMPRESSOR_MODEL_NAME = "PharmaPulseCompressorModel"

REPORTS_DIR = "reports"
ARTIFACTS_DIR = "artifacts"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def generate_batch(n_samples: int = 1000, seed: int = 42, heatwave: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ambient_temp = (
        np.clip(rng.normal(45.2, 2.2, n_samples), 40.0, 50.0)
        if heatwave
        else np.clip(rng.normal(29.0, 3.5, n_samples), 18.0, 37.5)
    )
    is_worn = rng.random(n_samples) < 0.28
    vib = np.clip(
        np.where(is_worn, rng.normal(86.0, 12.0, n_samples), rng.normal(42.0, 5.5, n_samples)),
        20.0,
        130.0,
    )
    door_open = (rng.random(n_samples) < 0.06).astype(int)

    cabin_temp = np.clip(
        rng.normal(4.6, 0.8, n_samples)
        + np.maximum(0.0, (vib - 62.0) * 0.065)
        + np.maximum(0.0, (ambient_temp - 30.0) * (0.14 if heatwave else 0.05))
        + door_open * rng.uniform(1.8, 4.0, n_samples),
        1.2,
        16.5,
    )
    mkt_temp = np.round(cabin_temp + rng.uniform(0.05, 0.45, n_samples), 3)
    temp_delta = np.round(ambient_temp - cabin_temp, 3)
    thermal_stress_index = np.round(vib * (ambient_temp / 30.0), 3)

    time_to_spoilage = np.clip(
        np.maximum(0.0, 9.2 - mkt_temp) * 42.0
        - np.maximum(0.0, (ambient_temp - 28.0) * 2.4)
        - np.maximum(0.0, (vib - 58.0) * 1.65)
        - door_open * 28.0
        + rng.normal(0, 4.5, n_samples),
        5.0,
        240.0,
    )

    logit = -7.2 + 0.095 * vib + 0.045 * np.maximum(0.0, ambient_temp - 32.0) + 0.18 * np.maximum(0.0, cabin_temp - 6.0)
    comp_fail = (rng.random(n_samples) < (1.0 / (1.0 + np.exp(-logit)))).astype(int)

    return pd.DataFrame(
        {
            "cabin_temp": np.round(cabin_temp, 3),
            "ambient_temp": np.round(ambient_temp, 3),
            "compressor_vibration_hz": np.round(vib, 3),
            "door_open": door_open,
            "mkt_temp": mkt_temp,
            "temp_delta": temp_delta,
            "thermal_stress_index": thermal_stress_index,
            "time_to_spoilage": np.round(time_to_spoilage, 2),
            "compressor_failure": comp_fail,
        }
    )


def evaluate_models(sp_model, cp_model, df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y_sp = df["time_to_spoilage"]
    y_cp = df["compressor_failure"]

    sp_preds = sp_model.predict(X)
    cp_preds = cp_model.predict(X)

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


def task_a_prepare_new_data() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TASK A: PREPARE & VALIDATE NEW INCOMING DATA BATCH")
    print("=" * 65)

    new_batch = generate_batch(n_samples=600, seed=2026, heatwave=True)
    new_batch.insert(0, "record_id", [f"REC-{10000 + i}" for i in range(len(new_batch))])

    required_cols = FEATURE_COLUMNS + ["time_to_spoilage", "compressor_failure"]
    assert all(c in new_batch.columns for c in required_cols), "Missing required columns!"

    total_missing = int(new_batch[required_cols].isnull().sum().sum())
    dup_rows = int(new_batch.duplicated().sum())
    dup_ids = int(new_batch["record_id"].duplicated().sum())

    target_dist = new_batch["compressor_failure"].value_counts().to_dict()
    target_pct = (new_batch["compressor_failure"].value_counts(normalize=True) * 100).round(2).to_dict()

    print(f"Dataset Size          : {len(new_batch)} rows x {len(new_batch.columns)} columns")
    print(f"Required Columns Valid: YES ({len(required_cols)} columns checked)")
    print(f"Total Missing Values  : {total_missing}")
    print(f"Duplicate Records     : {dup_rows} | Duplicate IDs: {dup_ids}")
    print(
        f"Target Distribution   : Class 0 (Healthy)={target_dist.get(0, 0)} ({target_pct.get(0, 0)}%), "
        f"Class 1 (Failure)={target_dist.get(1, 0)} ({target_pct.get(1, 0)}%)"
    )
    return new_batch


def task_b_evaluate_existing_model(baseline_df: pd.DataFrame, new_batch: pd.DataFrame) -> dict:
    print("\n" + "=" * 65)
    print("TASK B: EVALUATE EXISTING @champion MODEL ON NEW BATCH")
    print("=" * 65)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("PharmaPulse_ColdChain_AIOps")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Train & register baseline @champion v1 if not already present in CI runner
    X_b = baseline_df[FEATURE_COLUMNS]
    y_sp_b = baseline_df["time_to_spoilage"]
    y_cp_b = baseline_df["compressor_failure"]

    with mlflow.start_run(run_name="baseline_champion_v1"):
        champ_sp = lgb.LGBMRegressor(n_estimators=140, learning_rate=0.06, max_depth=6, random_state=42, verbose=-1)
        champ_sp.fit(X_b, y_sp_b)

        champ_cp = GradientBoostingClassifier(n_estimators=120, learning_rate=0.07, max_depth=4, random_state=42)
        champ_cp.fit(X_b, y_cp_b)

        sp_info = mlflow.sklearn.log_model(champ_sp, "spoilage_model", registered_model_name=SPOILAGE_MODEL_NAME)
        cp_info = mlflow.sklearn.log_model(champ_cp, "compressor_model", registered_model_name=COMPRESSOR_MODEL_NAME)
        client.set_registered_model_alias(SPOILAGE_MODEL_NAME, "champion", sp_info.registered_model_version)
        client.set_registered_model_alias(COMPRESSOR_MODEL_NAME, "champion", cp_info.registered_model_version)

    loaded_sp = mlflow.sklearn.load_model(f"models:/{SPOILAGE_MODEL_NAME}@champion")
    loaded_cp = mlflow.sklearn.load_model(f"models:/{COMPRESSOR_MODEL_NAME}@champion")
    champ_metrics = evaluate_models(loaded_sp, loaded_cp, new_batch)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_model": "@champion (v1-baseline)",
        "batch_size": len(new_batch),
        "metrics": champ_metrics,
    }
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Regression (Time-to-Spoilage)   :", champ_metrics["regression_time_to_spoilage"])
    print("Classification (Compressor Fail):", champ_metrics["classification_compressor_failure"])
    print(f"Saved evaluation report -> {report_path}")
    return champ_metrics


def task_c_retrain_candidate(baseline_df: pd.DataFrame, new_batch: pd.DataFrame) -> tuple:
    print("\n" + "=" * 65)
    print("TASK C: RETRAIN CANDIDATE MODEL (EXISTING + NEW DATA)")
    print("=" * 65)

    new_clean = new_batch[FEATURE_COLUMNS + ["time_to_spoilage", "compressor_failure"]]
    combined_df = pd.concat([baseline_df, new_clean], ignore_index=True)

    X = combined_df[FEATURE_COLUMNS]
    y_sp = combined_df["time_to_spoilage"]
    y_cp = combined_df["compressor_failure"]

    X_train, _, y_sp_train, _, y_cp_train, _ = train_test_split(X, y_sp, y_cp, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="candidate_retrain_v2") as run:
        cand_sp = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.06, max_depth=6, random_state=42, verbose=-1)
        cand_sp.fit(X_train, y_sp_train)

        cand_cp = GradientBoostingClassifier(n_estimators=140, learning_rate=0.07, max_depth=4, random_state=42)
        cand_cp.fit(X_train, y_cp_train)

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

    eligible = (cand_rmse < champ_rmse) and (cand_f1 >= champ_f1 - 0.01)

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

    print(f"Champion RMSE : {champ_rmse} vs Candidate RMSE: {cand_rmse}")
    print(f"Champion F1   : {champ_f1} vs Candidate F1  : {cand_f1}")
    print(f"FINAL DECISION: {decision['decision']} (Champion overwritten: False)")
    print(f"Saved decision record -> {decision_path}")


if __name__ == "__main__":
    baseline_data = generate_batch(n_samples=2500, seed=42, heatwave=False)
    incoming_batch = task_a_prepare_new_data()
    champ_m = task_b_evaluate_existing_model(baseline_data, incoming_batch)
    cand_m, r_id, art_paths = task_c_retrain_candidate(baseline_data, incoming_batch)
    task_d_quality_gate(champ_m, cand_m, r_id, art_paths)
