import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, ValidationError, field_validator

from data_generator import ensure_baseline_datasets
from mlops.drift_and_retrain import compute_drift_report, execute_retrain_and_promote
from mlops.mkt import FEATURE_COLUMNS, engineer_features_single
from mlops.train import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    configure_mlflow,
    rollback_to_v1,
    seed_initial_champion,
)

DRIFT_SCORE_GAUGE = Gauge("pharmapulse_feature_drift_score", "Current Evidently AI / KS feature drift score (0.0 to 1.0)")
QUARANTINED_PACKETS_COUNTER = Counter("pharmapulse_quarantined_packets_total", "Total corrupted telemetry packets rejected")
AUTONOMOUS_REROUTES_COUNTER = Counter("pharmapulse_autonomous_reroutes_total", "Total autonomous cold-storage hub reroutes")
CHAMPION_VERSION_GAUGE = Gauge("pharmapulse_champion_model_version", "Active @champion MLflow model version")


class TelemetryPacket(BaseModel):
    truck_id: str = Field(..., min_length=3, max_length=32)
    cabin_temp: float = Field(..., ge=-30.0, le=60.0)
    ambient_temp: float = Field(..., ge=-45.0, le=65.0)
    compressor_vibration_hz: float = Field(..., ge=0.0, le=250.0)
    door_open: int = Field(default=0, ge=0, le=1)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)

    @field_validator("cabin_temp", "ambient_temp", "compressor_vibration_hz", "lat", "lon")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Telemetry value must be finite")
        return v


def load_hubs() -> list[dict[str, Any]]:
    with Path("hubs.json").open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class PharmaPulseEngine:
    def __init__(self) -> None:
        self.hubs = load_hubs()
        self.baseline_df, _ = ensure_baseline_datasets()
        self.client: Optional[MlflowClient] = None
        self.champion_model = None
        self.champion_version: str = "1"
        self.champion_tag: str = "v1-baseline"
        self.challenger_model = None
        self.challenger_version: Optional[str] = None
        self.challenger_tag: Optional[str] = None
        self.heatwave_active: bool = False
        self.compressor_wear_truck: Optional[str] = None
        self.temp_windows: dict[str, deque[float]] = {}
        self.live_buffer: deque[dict[str, Any]] = deque(maxlen=250)
        self.quarantine_log: deque[dict[str, Any]] = deque(maxlen=40)
        self.remediation_log: deque[dict[str, Any]] = deque(maxlen=40)
        self.shadow_log: deque[dict[str, Any]] = deque(maxlen=30)
        self.quarantine_total: int = 0
        self.reroute_total: int = 0
        self.bay_counter: int = 100
        self.drift_report: dict[str, Any] = {"drift_score": 0.04, "drift_detected": False, "drifted_features": [], "method": "baseline_nominal"}
        DRIFT_SCORE_GAUGE.set(0.04)

        self.fleet: dict[str, dict[str, Any]] = {
            "TRK-101": {"truck_id": "TRK-101", "cargo": "mRNA Oncology Batch #A9", "route_name": "Bengaluru Airport -> Electronic City", "lat": 13.0827, "lon": 77.6910, "heading": 0.4, "cabin_temp": 4.5, "mkt_temp": 4.5, "ambient_temp": 27.5, "compressor_vibration_hz": 27.2, "door_open": 0, "time_to_spoilage": 295.0, "compressor_failure_prob": 0.03, "status": "NOMINAL", "reroute": None, "shadow": None},
            "TRK-102": {"truck_id": "TRK-102", "cargo": "Pediatric Hexavalent Vaccine #B4", "route_name": "Mumbai Port -> Navi Mumbai Hub", "lat": 19.0760, "lon": 72.8777, "heading": 1.8, "cabin_temp": 4.9, "mkt_temp": 4.9, "ambient_temp": 29.8, "compressor_vibration_hz": 29.4, "door_open": 0, "time_to_spoilage": 280.0, "compressor_failure_prob": 0.04, "status": "NOMINAL", "reroute": None, "shadow": None},
            "TRK-103": {"truck_id": "TRK-103", "cargo": "Insulin Glargine Cold-Pack #C2", "route_name": "Hyderabad ORR -> Genome Valley", "lat": 17.4850, "lon": 78.4867, "heading": 3.5, "cabin_temp": 5.1, "mkt_temp": 5.1, "ambient_temp": 30.2, "compressor_vibration_hz": 28.8, "door_open": 0, "time_to_spoilage": 272.0, "compressor_failure_prob": 0.05, "status": "NOMINAL", "reroute": None, "shadow": None},
            "TRK-104": {"truck_id": "TRK-104", "cargo": "Monoclonal Antibody Lot #D7", "route_name": "Gurugram -> IGI Cold-Chain Annex", "lat": 28.4950, "lon": 77.0890, "heading": 5.1, "cabin_temp": 4.7, "mkt_temp": 4.7, "ambient_temp": 26.4, "compressor_vibration_hz": 26.5, "door_open": 0, "time_to_spoilage": 302.0, "compressor_failure_prob": 0.02, "status": "NOMINAL", "reroute": None, "shadow": None},
        }
        for tid, tdata in self.fleet.items():
            self.temp_windows[tid] = deque([tdata["cabin_temp"]] * 6, maxlen=8)

    def load_models(self) -> dict[str, Any]:
        self.client = configure_mlflow()
        seed_initial_champion()
        champ_ver = self.client.get_model_version_by_alias(MODEL_NAME, "champion")
        self.champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
        self.champion_version = str(champ_ver.version)
        self.champion_tag = champ_ver.tags.get("model_tag", f"v{self.champion_version}")
        CHAMPION_VERSION_GAUGE.set(int(self.champion_version))
        try:
            chall_ver = self.client.get_model_version_by_alias(MODEL_NAME, "challenger")
            self.challenger_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@challenger")
            self.challenger_version = str(chall_ver.version)
            self.challenger_tag = chall_ver.tags.get("model_tag", f"v{self.challenger_version}")
        except Exception:
            self.challenger_model = None
            self.challenger_version = None
            self.challenger_tag = None
        return {"champion_version": self.champion_version, "champion_tag": self.champion_tag, "challenger_version": self.challenger_version, "challenger_tag": self.challenger_tag}

    def record_quarantine(self, raw_payload: dict[str, Any], err: ValidationError) -> dict[str, Any]:
        self.quarantine_total += 1
        QUARANTINED_PACKETS_COUNTER.inc()
        first_err = err.errors()[0] if err.errors() else {"loc": ("unknown",), "msg": str(err)}
        field_name = ".".join(str(x) for x in first_err.get("loc", ("unknown",)))
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "truck_id": str(raw_payload.get("truck_id", "UNKNOWN")),
            "field": field_name,
            "bad_value": str(raw_payload.get(field_name, "N/A")),
            "reason": first_err.get("msg", "Validation failed"),
        }
        self.quarantine_log.appendleft(entry)
        return entry

    def find_nearest_hub(self, lat: float, lon: float) -> tuple[dict[str, Any], float]:
        best_hub = self.hubs[0]
        best_dist = float("inf")
        for hub in self.hubs:
            dist = haversine_km(lat, lon, hub["lat"], hub["lon"])
            if dist < best_dist:
                best_dist = dist
                best_hub = hub
        return best_hub, round(best_dist, 2)

    def ingest_valid_packet(self, pkt: TelemetryPacket) -> dict[str, Any]:
        tid = pkt.truck_id
        if tid not in self.temp_windows:
            self.temp_windows[tid] = deque(maxlen=8)
        self.temp_windows[tid].append(pkt.cabin_temp)

        features = engineer_features_single(
            cabin_temp=pkt.cabin_temp,
            ambient_temp=pkt.ambient_temp,
            compressor_vibration_hz=pkt.compressor_vibration_hz,
            door_open=pkt.door_open,
            rolling_temps=list(self.temp_windows[tid]),
        )
        feat_df = pd.DataFrame([features])[FEATURE_COLUMNS]

        t0 = time.perf_counter()
        champ_out = self.champion_model.predict(feat_df).iloc[0]
        champ_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        tts_min = round(float(champ_out["time_to_spoilage"]), 1)
        fail_prob = round(float(champ_out["compressor_failure_prob"]), 3)

        shadow_info = None
        if self.challenger_model is not None:
            t1 = time.perf_counter()
            chall_out = self.challenger_model.predict(feat_df).iloc[0]
            chall_latency_ms = round((time.perf_counter() - t1) * 1000.0, 2)
            chall_tts = round(float(chall_out["time_to_spoilage"]), 1)
            chall_fail = round(float(chall_out["compressor_failure_prob"]), 3)
            shadow_info = {
                "challenger_version": self.challenger_version,
                "challenger_tts": chall_tts,
                "tts_delta": round(chall_tts - tts_min, 1),
                "challenger_fail_prob": chall_fail,
                "champion_latency_ms": champ_latency_ms,
                "challenger_latency_ms": chall_latency_ms,
            }
            self.shadow_log.appendleft({"timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"), "truck_id": tid, **shadow_info})

        reroute_target = None
        status = "NOMINAL"
        if tts_min < 45.0 or fail_prob > 0.75:
            status = "CRITICAL_REROUTE"
            existing_reroute = self.fleet.get(tid, {}).get("reroute")
            if existing_reroute is None:
                nearest_hub, dist_km = self.find_nearest_hub(pkt.lat, pkt.lon)
                self.bay_counter += 1
                bay_id = f"BAY-{self.bay_counter % 16 + 1:02d}"
                self.reroute_total += 1
                AUTONOMOUS_REROUTES_COUNTER.inc()
                trigger_reason = f"MKT Spoilage Countdown {tts_min}m (<45m)" if tts_min < 45.0 else f"Compressor Failure Prob {fail_prob:.2f} (>0.75)"
                reroute_target = {
                    "hub_id": nearest_hub["hub_id"],
                    "hub_name": nearest_hub["name"],
                    "hub_lat": nearest_hub["lat"],
                    "hub_lon": nearest_hub["lon"],
                    "distance_km": dist_km,
                    "reserved_bay": bay_id,
                    "trigger_reason": trigger_reason,
                }
                self.remediation_log.appendleft({
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    "truck_id": tid,
                    "action": f"Autonomous Reroute -> {nearest_hub['name']} ({bay_id})",
                    "distance_km": dist_km,
                    "reason": trigger_reason,
                })
            else:
                reroute_target = existing_reroute
        elif tts_min < 110.0 or fail_prob > 0.45 or features["mkt_temp"] > 7.5:
            status = "WARNING"

        existing = self.fleet.get(tid, {"truck_id": tid, "cargo": "Cold-Chain Biologics", "route_name": "Regional Transit", "heading": 0.0})
        existing.update({
            "lat": round(pkt.lat, 5),
            "lon": round(pkt.lon, 5),
            "cabin_temp": round(pkt.cabin_temp, 2),
            "mkt_temp": round(features["mkt_temp"], 2),
            "ambient_temp": round(pkt.ambient_temp, 2),
            "compressor_vibration_hz": round(pkt.compressor_vibration_hz, 2),
            "door_open": pkt.door_open,
            "time_to_spoilage": tts_min,
            "compressor_failure_prob": fail_prob,
            "status": status,
            "reroute": reroute_target,
            "shadow": shadow_info,
        })
        self.fleet[tid] = existing

        self.live_buffer.append({
            "truck_id": tid,
            "cabin_temp": pkt.cabin_temp,
            "ambient_temp": pkt.ambient_temp,
            "compressor_vibration_hz": pkt.compressor_vibration_hz,
            "door_open": pkt.door_open,
        })
        if len(self.live_buffer) % 8 == 0:
            self.refresh_drift()

        return {
            "accepted": True,
            "truck_id": tid,
            "status": status,
            "mkt_temp": round(features["mkt_temp"], 2),
            "time_to_spoilage": tts_min,
            "compressor_failure_prob": fail_prob,
            "champion_version": self.champion_version,
            "champion_latency_ms": champ_latency_ms,
            "shadow": shadow_info,
            "reroute": reroute_target,
        }

    def refresh_drift(self) -> dict[str, Any]:
        if len(self.live_buffer) < 5:
            return self.drift_report
        live_df = pd.DataFrame(list(self.live_buffer))
        self.drift_report = compute_drift_report(self.baseline_df, live_df)
        DRIFT_SCORE_GAUGE.set(self.drift_report["drift_score"])
        return self.drift_report

    def step_fleet_simulation(self) -> None:
        rng = np.random.default_rng()
        for tid, truck in self.fleet.items():
            if truck.get("reroute"):
                lat = truck["lat"] + 0.08 * (truck["reroute"]["hub_lat"] - truck["lat"])
                lon = truck["lon"] + 0.08 * (truck["reroute"]["hub_lon"] - truck["lon"])
            else:
                truck["heading"] = (truck.get("heading", 0.0) + 0.18) % (2 * math.pi)
                lat = truck["lat"] + 0.0025 * math.cos(truck["heading"])
                lon = truck["lon"] + 0.0025 * math.sin(truck["heading"])

            if self.heatwave_active:
                ambient = float(rng.uniform(45.0, 49.5))
            else:
                ambient = float(np.clip(truck["ambient_temp"] + rng.normal(0.0, 0.35), 24.0, 32.5))

            if self.compressor_wear_truck == tid:
                vib = float(rng.uniform(68.0, 82.0))
                cabin = float(np.clip(truck["cabin_temp"] + rng.uniform(0.55, 1.15), 4.5, 14.5))
            else:
                vib = float(np.clip(27.5 + rng.normal(0.0, 1.8), 21.0, 34.0))
                heat_leak = max(0.0, (ambient - 34.0) * 0.14)
                target_cabin = 4.7 + heat_leak
                cabin = float(truck["cabin_temp"] + 0.35 * (target_cabin - truck["cabin_temp"]) + rng.normal(0.0, 0.2))
                cabin = float(np.clip(cabin, 2.5, 11.5))

            pkt = TelemetryPacket(
                truck_id=tid,
                cabin_temp=round(cabin, 2),
                ambient_temp=round(ambient, 2),
                compressor_vibration_hz=round(vib, 2),
                door_open=0,
                lat=round(lat, 5),
                lon=round(lon, 5),
            )
            self.ingest_valid_packet(pkt)


engine = PharmaPulseEngine()


async def telemetry_loop() -> None:
    while True:
        try:
            engine.step_fleet_simulation()
        except Exception as exc:
            print(f"[TelemetryLoop Warning] {exc}")
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_models()
    for _ in range(6):
        engine.step_fleet_simulation()
    task = asyncio.create_task(telemetry_loop())
    yield
    task.cancel()


app = FastAPI(title="PharmaPulse AIOps & MLOps Platform", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "pharmapulse-aiops",
        "champion_version": engine.champion_version,
        "challenger_version": engine.challenger_version,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/api/state")
def get_system_state() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "champion_version": engine.champion_version,
            "champion_tag": engine.champion_tag,
            "challenger_version": engine.challenger_version,
            "challenger_tag": engine.challenger_tag,
        },
        "chaos": {"heatwave_active": engine.heatwave_active, "compressor_wear_truck": engine.compressor_wear_truck},
        "drift": engine.drift_report,
        "counters": {
            "quarantined_packets": engine.quarantine_total,
            "autonomous_reroutes": engine.reroute_total,
            "live_buffer_size": len(engine.live_buffer),
        },
        "hubs": engine.hubs,
        "fleet": list(engine.fleet.values()),
        "quarantine_log": list(engine.quarantine_log),
        "remediation_log": list(engine.remediation_log),
        "shadow_log": list(engine.shadow_log),
    }


@app.post("/api/telemetry")
def ingest_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        packet = TelemetryPacket.model_validate(payload)
    except ValidationError as exc:
        q_entry = engine.record_quarantine(payload, exc)
        return {"accepted": False, "quarantined": True, "quarantine_entry": q_entry, "total_quarantined": engine.quarantine_total}
    return engine.ingest_valid_packet(packet)


@app.post("/api/chaos/compressor-wear")
def chaos_compressor_wear(truck_id: str = "TRK-101") -> dict[str, Any]:
    if truck_id not in engine.fleet:
        truck_id = next(iter(engine.fleet.keys()))
    engine.compressor_wear_truck = truck_id
    t = engine.fleet[truck_id]
    pkt = TelemetryPacket(truck_id=truck_id, cabin_temp=min(13.8, t["cabin_temp"] + 4.8), ambient_temp=t["ambient_temp"], compressor_vibration_hz=79.5, door_open=0, lat=t["lat"], lon=t["lon"])
    return {"status": "injected_compressor_wear", "truck_id": truck_id, "result": engine.ingest_valid_packet(pkt)}


@app.post("/api/chaos/heatwave")
def chaos_heatwave_drift() -> dict[str, Any]:
    engine.heatwave_active = True
    rng = np.random.default_rng(77)
    tids = list(engine.fleet.keys())
    for i in range(18):
        tid = tids[i % len(tids)]
        t = engine.fleet[tid]
        pkt = TelemetryPacket(truck_id=tid, cabin_temp=round(float(rng.uniform(7.2, 10.6)), 2), ambient_temp=round(float(rng.uniform(45.5, 49.8)), 2), compressor_vibration_hz=round(float(rng.uniform(34.0, 48.0)), 2), door_open=0, lat=t["lat"], lon=t["lon"])
        engine.ingest_valid_packet(pkt)
    return {"status": "heatwave_drift_active", "drift": engine.refresh_drift()}


@app.post("/api/chaos/corrupt-packet")
def chaos_corrupt_packet(truck_id: str = "TRK-102") -> dict[str, Any]:
    return ingest_telemetry({"truck_id": truck_id, "cabin_temp": -99.0, "ambient_temp": 31.0, "compressor_vibration_hz": -12.5, "door_open": 0, "lat": 19.0760, "lon": 72.8777})


@app.post("/api/retrain")
def trigger_retrain() -> dict[str, Any]:
    live_df = pd.DataFrame(list(engine.live_buffer)) if engine.live_buffer else None
    result = execute_retrain_and_promote(live_df=live_df, reload_api_url=None)
    if result["promoted"]:
        engine.load_models()
    return result


@app.post("/reload-model")
def reload_model() -> dict[str, Any]:
    return {"status": "reloaded", **engine.load_models()}


@app.post("/rollback")
def rollback_model() -> dict[str, Any]:
    rb = rollback_to_v1()
    engine.heatwave_active = False
    engine.compressor_wear_truck = None
    engine.live_buffer.clear()
    for tid, truck in engine.fleet.items():
        truck["cabin_temp"] = 4.8
        truck["mkt_temp"] = 4.8
        truck["ambient_temp"] = 28.0
        truck["compressor_vibration_hz"] = 27.5
        truck["status"] = "NOMINAL"
        truck["reroute"] = None
        engine.temp_windows[tid] = deque([4.8] * 6, maxlen=8)
    engine.drift_report = {"drift_score": 0.03, "drift_detected": False, "drifted_features": [], "method": "reset_to_baseline_v1"}
    DRIFT_SCORE_GAUGE.set(0.03)
    return {**rb, **engine.load_models()}


static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_spa():
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index_path)
