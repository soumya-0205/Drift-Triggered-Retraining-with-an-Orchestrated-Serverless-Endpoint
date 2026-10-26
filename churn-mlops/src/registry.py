"""Thin MLflow wrapper. SQLite backend + local artifact store = free, no server needed.
The *only* way a model reaches production is promote(), which the workflow calls after the gate passes."""
import json, os, mlflow, mlflow.sklearn
from pathlib import Path
from mlflow.tracking import MlflowClient
from common import ROOT

MODEL_NAME, PROD_ALIAS = "churn-model", "production"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT.as_posix()}/mlflow.db")
EXPERIMENT = "churn-drift-retraining"

def _artifact_uri():
    """Always a proper file:// URI (plain Windows paths like C:\\x break MLflow)."""
    p = os.environ.get("ARTIFACT_ROOT", str(ROOT / "mlartifacts"))
    return p if p.startswith("file:") else Path(p).resolve().as_uri()

def _setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=_artifact_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient(TRACKING_URI)

def register(model, profile, metrics, params, tags=None, run_name="train"):
    """Log run, store model + evaluation + reference profile, create a new registered version. Does NOT promote."""
    client = _setup()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params); mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(profile, "reference_profile.json"); mlflow.log_dict(metrics, "evaluation.json")
        info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, mv.version, "run_id", run.info.run_id)
        for k, v in (tags or {}).items(): client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
    return int(mv.version)

def promote(version):
    client = _setup(); client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, str(version)); return int(version)

def production_version():
    client = _setup(); return int(client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS).version)

def load_production():
    """-> (version, sklearn pipeline, reference_profile dict)"""
    client = _setup(); mv = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
    return int(mv.version), load_version(mv.version)[0], load_version(mv.version)[1]

def load_version(version):
    client = _setup(); mv = client.get_model_version(MODEL_NAME, str(version))
    run_id = mv.tags["run_id"]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    prof = json.load(open(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference_profile.json")))
    return model, prof
