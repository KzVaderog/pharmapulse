import subprocess
import sys
from typing import Optional
import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from data_generator import ensure_baseline_datasets, generate_cold_chain_dataset
from mlops.mkt import engineer_features_df
from mlops.train import configure_mlflow, train_and_register_model, MODEL_NAME

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False

DRIFT_MONITOR_COLS = [
    "cabin_temp",
    "ambient_temp",
    "compressor_vibration_hz",
    "mkt_temp",
    "temp_delta",
    "thermal_stress_index",
]


def compute_drift_report(baseline_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    if live_df is None or len(live_df) < 5:
        return {"drift_score": 0.04, "drift_detected": False, "drifted_features": [], "method": "insufficient_live_samples"}

    ref = baseline_df[DRIFT_MONITOR_COLS].copy()
    cur = engineer_features_df(live_df)[DRIFT_MONITOR_COLS].copy()

    drifted_features: list[str] = []
    ks_stats: list[float] = []
    for col in DRIFT_MONITOR_COLS:
        stat, p_val = ks_2samp(ref[col].to_numpy(), cur[col].to_numpy())
        ks_stats.append(float(stat))
        if p_val < 0.01 and stat > 0.25:
            drifted_features.append(col)

    evidently_share = None
    method = "scipy_ks_statistical"
    if EVIDENTLY_AVAILABLE and len(cur) >= 10:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref.iloc[:500], current_data=cur.iloc[-200:])
            evidently_share = float(report.as_dict()["metrics"][0]["result"].get("share_of_drifted_columns", 0.0))
            method = "evidently_data_drift_preset"
        except Exception:
            evidently_share = None

    mean_ks = float(np.mean(ks_stats))
    drift_score = round(0.6 * evidently_share + 0.4 * mean_ks, 4) if evidently_share is not None else round(mean_ks, 4)
    drift_score = float(np.clip(drift_score, 0.0, 1.0))
    return {
        "drift_score": drift_score,
        "drift_detected": bool(drift_score >= 0.35 or len(drifted_features) >= 2),
        "drifted_features": drifted_features,
        "method": method,
        "sample_count": int(len(cur)),
    }


def execute_retrain_and_promote(
    live_df: Optional[pd.DataFrame] = None,
    reload_api_url: Optional[str] = "http://127.0.0.1:8000/reload-model",
) -> dict:
    client = configure_mlflow()
    base_df, _ = ensure_baseline_datasets()
    heatwave_train_df = generate_cold_chain_dataset(n_samples=1600, heatwave_ratio=0.75, seed=2026)
    frames = [base_df, heatwave_train_df]
    if live_df is not None and len(live_df) > 0:
        frames.append(engineer_features_df(live_df))
    combined_df = pd.concat(frames, ignore_index=True)

    train_res = train_and_register_model(df=combined_df, alias="challenger", model_tag="v2-heatwave-adapted", seed=2026)
    challenger_version = train_res["version"]

    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_quality_gates.py", "-v"], capture_output=True, text=True)
    promoted = proc.returncode == 0
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)
        if reload_api_url:
            try:
                requests.post(reload_api_url, timeout=3.0)
            except Exception:
                pass

    return {
        "status": "promoted" if promoted else "rejected_by_quality_gates",
        "promoted": promoted,
        "challenger_version": challenger_version,
        "champion_version": challenger_version if promoted else "1",
        "metrics": train_res,
        "pytest_exit_code": proc.returncode,
        "pytest_output": proc.stdout[-2500:],
    }
