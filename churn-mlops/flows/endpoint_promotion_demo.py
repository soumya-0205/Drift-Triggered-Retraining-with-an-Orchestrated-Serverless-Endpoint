"""Runs a REAL local HTTP endpoint (functions-framework) while the weekly workflow runs, and probes it after every week.
Shows the served model_version change after promotion with no restart and no code change. Isolated registry."""
import os, sys, json, time, subprocess, tempfile, urllib.request
from pathlib import Path
T = Path(tempfile.mkdtemp(prefix="demo_")); ROOT = Path(__file__).resolve().parents[1]
os.environ.update(MLFLOW_TRACKING_URI=f"sqlite:///{T.as_posix()}/mlflow.db", ARTIFACT_ROOT=str(T / "art"), PIPELINE_STATE_DIR=str(T / "state"),
                  PIPELINE_REPORT_DIR=str(T / "reports"), HUMAN_DECISIONS=str(T / "human.json"), REFRESH_SECONDS="0.5")
sys.path.insert(0, str(ROOT / "function")); (T / "human.json").write_text("{}")
from weekly_pipeline import *
import train_baseline; train_baseline.main()
srv = subprocess.Popen([sys.executable, "-m", "functions_framework", "--source", str(ROOT / "function" / "main.py"), "--target", "predict", "--port", "8085"],
                       env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PROBE = {"instances": [{"Tenure_Months": 5, "Contract_Type": "Month-to-month", "Internet_Service": "Fiber", "Monthly_Charges": 85.0,
                        "Total_Charges": 400.0, "Support_Calls_Last_Month": 3, "Avg_Data_Usage_GB": 120.0, "Late_Payments": 1}]}
def call(payload=PROBE):
    req = urllib.request.Request("http://127.0.0.1:8085/", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try: return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except urllib.error.HTTPError as e: return {"http_error": e.code, "body": e.read().decode()}
for _ in range(40):
    try: call(); break
    except Exception: time.sleep(0.5)
print("ENDPOINT START:", call())
print("BAD INPUT (missing feature):", call({"instances": [{"Tenure_Months": 5}]}))
for w in range(1, 9):
    S = weekly_pipeline(w, run_tag="demo")
    if S["escalated"]:
        (T / "human.json").write_text(json.dumps({str(w): {"decision": "approve_retrain", "by": "simulated on-call", "note": "demo"}}))
    time.sleep(0.8); r = call()
    print(f"AFTER WEEK {w}: workflow prod=v{S['prod_version_after']} status={S['status']} | endpoint serves model_version={r['model_version']} p={r['predictions'][0]['churn_probability']}")
srv.terminate()
