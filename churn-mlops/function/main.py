"""Task 2: serverless-style prediction endpoint (Google Cloud Functions / Lambda-shaped handler, run locally with functions-framework).

POST {"instances":[{feature: value, ...}, ...]}  ->  {"model_version":int, "threshold":float, "predictions":[{"churn_probability":p,"churn_label":0/1}]}
The model is resolved from the registry alias 'production' - promotion changes what is served with NO code change or redeploy.
Warm instances re-check the alias at most every REFRESH_SECONDS (cheap metadata call) and reload only if the version changed."""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import functions_framework
import pandas as pd
import registry
from common import FEATURES, NUM, CAT, THRESHOLD

REFRESH_SECONDS = float(os.environ.get("REFRESH_SECONDS", "2"))
_cache = {"version": None, "model": None, "checked": 0.0}

def _model():
    now = time.time()
    if _cache["model"] is None or now - _cache["checked"] > REFRESH_SECONDS:
        v = registry.production_version()
        if v != _cache["version"]:
            _cache["version"], _cache["model"] = v, registry.load_version(v)[0]
        _cache["checked"] = now
    return _cache["version"], _cache["model"]

def _validate(instances):
    if not isinstance(instances, list) or not instances: raise ValueError("'instances' must be a non-empty list")
    df = pd.DataFrame(instances)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError(f"missing features: {missing}")
    for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce")      # missing numeric -> NaN -> median imputer handles it
    return df[FEATURES]

@functions_framework.http
def predict(request):
    try:
        body = request.get_json(silent=True) or {}
        X = _validate(body.get("instances"))
    except ValueError as e:
        return (json.dumps({"error": str(e)}), 400, {"Content-Type": "application/json"})
    try:
        version, model = _model()
    except Exception as e:                       # registry/model unavailable -> serve last good model if we have one
        if _cache["model"] is None: return (json.dumps({"error": f"model unavailable: {e}"}), 503, {"Content-Type": "application/json"})
        version, model = _cache["version"], _cache["model"]
    p = model.predict_proba(X)[:, 1]
    out = [{"churn_probability": round(float(x), 4), "churn_label": int(x >= THRESHOLD)} for x in p]
    return (json.dumps({"model_version": version, "threshold": THRESHOLD, "predictions": out}), 200, {"Content-Type": "application/json"})
