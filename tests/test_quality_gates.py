import time
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from data_generator import ensure_baseline_datasets
from mlops.mkt import FEATURE_COLUMNS
from mlops.train import MLFLOW_TRACKING_URI, MODEL_NAME, configure_mlflow, seed_initial_champion


@pytest.fixture(scope="module")
def ml_context():
    configure_mlflow()
    seed_initial_champion()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    base_df, heatwave_df = ensure_baseline_datasets()
    return client, base_df, heatwave_df


def test_gate_1_data_schema_and_bounds_integrity(ml_context):
    _, base_df, heatwave_df = ml_context
    for name, df in [("baseline", base_df), ("heatwave", heatwave_df)]:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        assert not missing
        assert not df[FEATURE_COLUMNS].isnull().any().any()
        assert df["cabin_temp"].between(-30.0, 60.0).all()
        assert df["ambient_temp"].between(-40.0, 65.0).all()
        assert (df["compressor_vibration_hz"] >= 0.0).all()
        assert df["time_to_spoilage"].between(1.0, 360.0).all()


def test_gate_2_challenger_beats_champion_on_heatwave(ml_context):
    client, _, heatwave_df = ml_context
    champion_ver = client.get_model_version_by_alias(MODEL_NAME, "champion")
    try:
        challenger_ver = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    except Exception:
        pytest.skip("No @challenger model registered yet.")

    champ_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    chall_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@challenger")
    X_val = heatwave_df[FEATURE_COLUMNS]
    y_true = heatwave_df["time_to_spoilage"].to_numpy()

    champ_rmse = float(np.sqrt(mean_squared_error(y_true, champ_model.predict(X_val)["time_to_spoilage"].to_numpy())))
    chall_rmse = float(np.sqrt(mean_squared_error(y_true, chall_model.predict(X_val)["time_to_spoilage"].to_numpy())))
    print(f"\nChampion v{champion_ver.version} RMSE: {champ_rmse:.3f} | Challenger v{challenger_ver.version} RMSE: {chall_rmse:.3f}")
    assert chall_rmse < champ_rmse


def test_gate_3_p95_inference_latency_under_20ms(ml_context):
    _, _, heatwave_df = ml_context
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@challenger")
    except Exception:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

    sample_rows = [heatwave_df[FEATURE_COLUMNS].iloc[[i % len(heatwave_df)]] for i in range(60)]
    for row in sample_rows[:5]:
        model.predict(row)

    latencies_ms = []
    for row in sample_rows[5:]:
        t0 = time.perf_counter()
        model.predict(row)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p95_ms = float(np.percentile(latencies_ms, 95))
    print(f"\nSingle-packet p95 latency: {p95_ms:.2f} ms")
    assert p95_ms < 20.0
