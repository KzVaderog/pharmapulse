import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uvicorn

from mlops.train import MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI, seed_initial_champion


def main() -> None:
    Path(MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    seed_info = seed_initial_champion()
    print(f"[PharmaPulse] Registry ready: {seed_info}")

    mlflow_cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", MLFLOW_TRACKING_URI,
        "--default-artifact-root", Path(MLFLOW_ARTIFACT_ROOT).as_uri(),
        "--host", "0.0.0.0", "--port", "5000",
    ]
    mlflow_proc = subprocess.Popen(mlflow_cmd)

    def shutdown_handler(signum, frame):
        mlflow_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    time.sleep(1.5)

    try:
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, log_level="info")
    finally:
        mlflow_proc.terminate()
        mlflow_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
