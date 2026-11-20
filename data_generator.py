from pathlib import Path
import numpy as np
import pandas as pd
from mlops.mkt import calculate_mkt, engineer_features_df


def generate_cold_chain_dataset(
    n_samples: int = 2400,
    heatwave_ratio: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_heatwave = int(n_samples * heatwave_ratio)
    n_normal = n_samples - n_heatwave
    records = []
    truck_ids = ["TRK-101", "TRK-102", "TRK-103", "TRK-104"]

    for idx in range(n_samples):
        is_heatwave = idx >= n_normal
        truck_id = truck_ids[idx % len(truck_ids)]
        ambient_temp = float(rng.uniform(42.5, 50.5)) if is_heatwave else float(rng.uniform(18.0, 33.5))
        regime_roll = rng.random()
        door_open = int(rng.random() < 0.08)

        if regime_roll < 0.65:
            compressor_vibration_hz = float(rng.normal(28.0, 3.2))
            heat_leak = max(0.0, (ambient_temp - 30.0) * 0.09)
            cabin_temp = float(rng.normal(4.6 + heat_leak + (1.4 if door_open else 0.0), 0.7))
        elif regime_roll < 0.85:
            compressor_vibration_hz = float(rng.uniform(54.0, 82.0))
            wear_factor = (compressor_vibration_hz - 50.0) / 30.0
            heat_leak = max(0.0, (ambient_temp - 25.0) * 0.16)
            cabin_temp = float(rng.normal(6.2 + 3.8 * wear_factor + heat_leak + (1.8 if door_open else 0.0), 0.9))
        else:
            compressor_vibration_hz = float(rng.uniform(42.0, 75.0))
            cabin_temp = float(rng.uniform(8.2, 14.5) + (max(0.0, ambient_temp - 38.0) * 0.22))

        compressor_vibration_hz = float(np.clip(compressor_vibration_hz, 12.0, 95.0))
        cabin_temp = float(np.clip(cabin_temp, 0.5, 24.0))

        hist_noise = rng.normal(-0.35, 0.45, size=5)
        window = [float(np.clip(cabin_temp + d, 0.5, 24.0)) for d in hist_noise] + [cabin_temp]
        mkt_temp = calculate_mkt(window)

        thermal_excess = max(0.0, mkt_temp - 5.5)
        ambient_penalty = max(0.0, ambient_temp - 32.0) * 1.45
        wear_penalty = max(0.0, compressor_vibration_hz - 45.0) * 1.35
        door_penalty = 18.0 if door_open else 0.0

        raw_tts = (
            310.0
            - (thermal_excess ** 1.65) * 26.0
            - ambient_penalty
            - wear_penalty
            - door_penalty
            + float(rng.normal(0.0, 6.5))
        )
        time_to_spoilage = float(np.clip(raw_tts, 5.0, 360.0))

        failure_logit = (
            -5.2
            + 0.095 * (compressor_vibration_hz - 35.0)
            + 0.065 * max(0.0, ambient_temp - 30.0)
            + 0.14 * max(0.0, cabin_temp - 6.5)
        )
        failure_prob = 1.0 / (1.0 + np.exp(-failure_logit))
        compressor_failure = int(failure_prob > 0.5)

        records.append({
            "truck_id": truck_id,
            "cabin_temp": round(cabin_temp, 3),
            "ambient_temp": round(ambient_temp, 3),
            "compressor_vibration_hz": round(compressor_vibration_hz, 3),
            "door_open": door_open,
            "mkt_temp": round(mkt_temp, 3),
            "time_to_spoilage": round(time_to_spoilage, 2),
            "compressor_failure": compressor_failure,
        })

    return engineer_features_df(pd.DataFrame(records))


def ensure_baseline_datasets(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    baseline_path = path / "baseline_telemetry.csv"
    heatwave_path = path / "heatwave_validation.csv"

    if not baseline_path.exists():
        df_base = generate_cold_chain_dataset(n_samples=2000, heatwave_ratio=0.0, seed=42)
        df_base.to_csv(baseline_path, index=False)
    else:
        df_base = pd.read_csv(baseline_path)

    if not heatwave_path.exists():
        df_heat = generate_cold_chain_dataset(n_samples=800, heatwave_ratio=0.85, seed=108)
        df_heat.to_csv(heatwave_path, index=False)
    else:
        df_heat = pd.read_csv(heatwave_path)

    return df_base, df_heat
