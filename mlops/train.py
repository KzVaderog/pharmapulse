import os
from pathlib import Path
import joblib
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from data_generator import ensure_baseline_datasets
from mlops.mkt import FEATURE_COLUMNS

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", str(Path("./mlruns").resolve()))
EXPERIMENT_NAME = "PharmaPulse-ColdChain-Spoilage"
MODEL_NAME = "PharmaPulseSpoilageModel"


class PharmaPulseDualPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, regressor=None, classifier=None, bundle_path=None):
        self.regressor = regressor
        self.classifier = classifier
        self.bundle_path = bundle_path
        self.feature_columns = list(FEATURE_COLUMNS)

    def _ensure_loaded(self):
        if self.regressor is not None and self.classifier is not None:
            return
        if self.bundle_path and Path(self.bundle_path).exists():
            bundle = joblib.load(self.bundle_path)
            self.regressor = bundle["regressor"]
            self.classifier = bundle["classifier"]
            self.feature_columns = bundle.get("feature_columns", list(FEATURE_COLUMNS))
            return
        art_dir = Path("data/artifacts")
        bundles = sorted(art_dir.glob("bundle_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if bundles:
            bundle = joblib.load(bundles[0])
            self.regressor = bundle["regressor"]
            self.classifier = bundle["classifier"]
            self.feature_columns = bundle.get("feature_columns", list(FEATURE_COLUMNS))

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["model_bundle"])
        self.regressor = bundle["regressor"]
        self.classifier = bundle["classifier"]
        self.feature_columns = bundle.get("feature_columns", list(FEATURE_COLUMNS))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        cols = getattr(self, "feature_columns", list(FEATURE_COLUMNS))
        X = model_input[cols]
        tts_pred = np.clip(self.regressor.predict(X), 1.0, 360.0)
        fail_prob = np.clip(self.classifier.predict_proba(X)[:, 1], 0.0, 1.0)
        return pd.DataFrame(
            {
                "time_to_spoilage": tts_pred,
                "compressor_failure_prob": fail_prob,
            }
        )


def configure_mlflow() -> MlflowClient:
    Path(MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=Path(MLFLOW_ARTIFACT_ROOT).as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return client


def train_and_register_model(
    df: pd.DataFrame,
    alias: str = "champion",
    model_tag: str = "v1-baseline",
    seed: int = 42,
) -> dict:
    client = configure_mlflow()
    _, heatwave_val = ensure_baseline_datasets()

    X = df[FEATURE_COLUMNS]
    y_reg = df["time_to_spoilage"]
    y_clf = df["compressor_failure"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=seed
    )

    regressor = LGBMRegressor(
        n_estimators=160, learning_rate=0.06, num_leaves=31, random_state=seed, verbose=-1, n_jobs=1
    )
    regressor.fit(X_train, y_reg_train)

    classifier = GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.07, max_depth=3, random_state=seed
    )
    classifier.fit(X_train, y_clf_train)

    in_dist_rmse = float(np.sqrt(mean_squared_error(y_reg_test, regressor.predict(X_test))))
    in_dist_auc = (
        float(roc_auc_score(y_clf_test, classifier.predict_proba(X_test)[:, 1]))
        if len(set(y_clf_test)) > 1
        else 1.0
    )
    heatwave_rmse = float(
        np.sqrt(mean_squared_error(heatwave_val["time_to_spoilage"], regressor.predict(heatwave_val[FEATURE_COLUMNS])))
    )

    bundle_dir = Path("data/artifacts")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"bundle_{model_tag}.joblib"
    joblib.dump(
        {"regressor": regressor, "classifier": classifier, "feature_columns": list(FEATURE_COLUMNS)},
        bundle_path,
    )

    with mlflow.start_run(run_name=f"train-{model_tag}") as run:
        mlflow.log_param("model_tag", model_tag)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metric("in_dist_rmse", in_dist_rmse)
        mlflow.log_metric("in_dist_auc", in_dist_auc)
        mlflow.log_metric("heatwave_rmse", heatwave_rmse)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PharmaPulseDualPyFunc(
                regressor=regressor,
                classifier=classifier,
                bundle_path=str(bundle_path),
            ),
            artifacts={"model_bundle": str(bundle_path)},
            input_example=X_train.iloc[:3].to_dict(orient="records"),
            registered_model_name=MODEL_NAME,
        )

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    version_num = str(max(versions, key=lambda v: int(v.version)).version)
    client.set_model_version_tag(MODEL_NAME, version_num, "model_tag", model_tag)
    client.set_model_version_tag(MODEL_NAME, version_num, "heatwave_rmse", f"{heatwave_rmse:.4f}")
    client.set_registered_model_alias(MODEL_NAME, alias, version_num)

    return {
        "run_id": run.info.run_id,
        "version": version_num,
        "alias": alias,
        "model_tag": model_tag,
        "in_dist_rmse": round(in_dist_rmse, 4),
        "in_dist_auc": round(in_dist_auc, 4),
        "heatwave_rmse": round(heatwave_rmse, 4),
    }


def seed_initial_champion() -> dict:
    client = configure_mlflow()
    base_df, _ = ensure_baseline_datasets()
    try:
        existing = client.get_model_version_by_alias(MODEL_NAME, "champion")
        return {"status": "already_seeded", "version": str(existing.version), "alias": "champion"}
    except Exception:
        res = train_and_register_model(df=base_df, alias="champion", model_tag="v1-baseline", seed=42)
        return {"status": "seeded", **res}


def rollback_to_v1() -> dict:
    client = configure_mlflow()
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")
    try:
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
    except Exception:
        pass
    return {"status": "rolled_back", "champion_version": "1"}
