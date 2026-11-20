# PharmaPulse — Cold-Chain AIOps & MLOps Platform

[![PharmaPulse CI](https://github.com/KzVaderog/pharmapulse/actions/workflows/ci.yml/badge.svg)](https://github.com/KzVaderog/pharmapulse/actions/workflows/ci.yml)

Dockerized cold-chain pharmaceutical spoilage prevention and fleet telemetry AIOps/MLOps platform built with **Python 3.11**, **FastAPI**, **MLflow**, **LightGBM**, **Scikit-learn**, **Evidently AI**, and **Prometheus**.

## Architecture & Modules
- `backend/main.py`: FastAPI telemetry ingestion, Pydantic v2 validation & quarantine feed, autonomous cold-hub rerouting, drift/retrain/rollback endpoints, and `/metrics`.
- `mlops/mkt.py`: Mean Kinetic Temperature (MKT) calculation and cold-chain feature engineering.
- `mlops/train.py`: LightGBM + GradientBoosting dual PyFunc model training, MLflow SQLite registry seeding (`@champion` / `@challenger`), and rollback.
- `mlops/drift_and_retrain.py`: Feature drift detection and retraining/promotion gates.
- `static/index.html`: Live spatial command center UI.
- `tests/test_quality_gates.py`: Schema/bounds integrity, challenger-vs-champion evaluation, and p95 inference latency quality gates.

## Quick Start (Docker Compose)
```bash