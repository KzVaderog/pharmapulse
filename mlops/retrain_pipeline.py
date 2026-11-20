import json
from mlops.drift_and_retrain import execute_retrain_and_promote
from mlops.train import seed_initial_champion


def main() -> None:
    seed_initial_champion()
    result = execute_retrain_and_promote(reload_api_url="http://127.0.0.1:8000/reload-model")
    print(json.dumps({k: v for k, v in result.items() if k != "pytest_output"}, indent=2))
    print("\n--- PYTEST QUALITY GATE OUTPUT ---")
    print(result["pytest_output"])
    if not result["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
