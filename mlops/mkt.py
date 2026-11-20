from typing import Sequence
import numpy as np
import pandas as pd

DELTA_H_KJ_MOL: float = 83.14472
GAS_CONSTANT_KJ_MOL_K: float = 8.314472e-3
ACTIVATION_RATIO: float = DELTA_H_KJ_MOL / GAS_CONSTANT_KJ_MOL_K
KELVIN_OFFSET: float = 273.15

FEATURE_COLUMNS: list[str] = [
    "cabin_temp",
    "ambient_temp",
    "compressor_vibration_hz",
    "door_open",
    "mkt_temp",
    "temp_delta",
    "thermal_stress_index",
    "vibration_thermal_load",
    "excursion_above_8c",
]


def calculate_mkt(temps_celsius: Sequence[float]) -> float:
    if temps_celsius is None or len(temps_celsius) == 0:
        return 5.0
    arr = np.clip(np.asarray(temps_celsius, dtype=np.float64), -50.0, 100.0)
    temps_kelvin = arr + KELVIN_OFFSET
    exponents = -ACTIVATION_RATIO / temps_kelvin
    max_exp = np.max(exponents)
    mean_exp = np.exp(max_exp) * np.mean(np.exp(exponents - max_exp))
    mkt_kelvin = -ACTIVATION_RATIO / np.log(mean_exp)
    return float(mkt_kelvin - KELVIN_OFFSET)


def engineer_features_single(
    cabin_temp: float,
    ambient_temp: float,
    compressor_vibration_hz: float,
    door_open: int,
    rolling_temps: Sequence[float],
) -> dict[str, float]:
    window = list(rolling_temps) if rolling_temps else [cabin_temp]
    mkt_c = calculate_mkt(window)
    temp_delta = ambient_temp - cabin_temp
    thermal_stress_index = max(0.0, mkt_c - 5.0) * (1.0 + 0.04 * max(0.0, temp_delta))
    vibration_thermal_load = compressor_vibration_hz * max(1.0, ambient_temp / 25.0)
    excursion_above_8c = max(0.0, mkt_c - 8.0)
    return {
        "cabin_temp": float(cabin_temp),
        "ambient_temp": float(ambient_temp),
        "compressor_vibration_hz": float(compressor_vibration_hz),
        "door_open": float(int(door_open)),
        "mkt_temp": float(mkt_c),
        "temp_delta": float(temp_delta),
        "thermal_stress_index": float(thermal_stress_index),
        "vibration_thermal_load": float(vibration_thermal_load),
        "excursion_above_8c": float(excursion_above_8c),
    }


def engineer_features_df(df: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
    out = df.copy()
    if "mkt_temp" not in out.columns:
        if "truck_id" in out.columns:
            mkt_values = np.zeros(len(out), dtype=np.float64)
            for _, group in out.groupby("truck_id", sort=False):
                temps = group["cabin_temp"].to_numpy()
                idxs = group.index.to_numpy()
                for i in range(len(temps)):
                    start = max(0, i - window_size + 1)
                    mkt_values[idxs[i]] = calculate_mkt(temps[start : i + 1])
            out["mkt_temp"] = mkt_values
        else:
            temps = out["cabin_temp"].to_numpy()
            out["mkt_temp"] = [
                calculate_mkt(temps[max(0, i - window_size + 1) : i + 1])
                for i in range(len(temps))
            ]
    out["door_open"] = out["door_open"].astype(float)
    out["temp_delta"] = out["ambient_temp"] - out["cabin_temp"]
    out["thermal_stress_index"] = np.maximum(0.0, out["mkt_temp"] - 5.0) * (
        1.0 + 0.04 * np.maximum(0.0, out["temp_delta"])
    )
    out["vibration_thermal_load"] = out["compressor_vibration_hz"] * np.maximum(
        1.0, out["ambient_temp"] / 25.0
    )
    out["excursion_above_8c"] = np.maximum(0.0, out["mkt_temp"] - 8.0)
    return out
