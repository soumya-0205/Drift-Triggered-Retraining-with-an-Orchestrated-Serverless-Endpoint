"""Task 6: fault injection in an ISOLATED registry/state (never touches the main run). After every fault we check that
production version and the endpoint's prediction are unchanged (except the flaky-registry case, which must self-heal)."""
import os, sys, json, shutil, tempfile
from pathlib import Path
T = Path(tempfile.mkdtemp(prefix="faults_"))
os.environ.update(MLFLOW_TRACKING_URI=f"sqlite:///{T.as_posix()}/mlflow.db", ARTIFACT_ROOT=str(T / "art"), PIPELINE_STATE_DIR=str(T / "state"),
                  PIPELINE_REPORT_DIR=str(T / "reports"), HUMAN_DECISIONS=str(T / "human.json"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "function"))
from weekly_pipeline import *
import train_baseline, functions_framework
train_baseline.main()
client = functions_framework.create_app(target="predict", source=str(Path(__file__).resolve().parents[1] / "function" / "main.py")).test_client()
PROBE = {"instances": [{"Tenure_Months": 5, "Contract_Type": "Month-to-month", "Internet_Service": "Fiber", "Monthly_Charges": 85.0,
                        "Total_Charges": 400.0, "Support_Calls_Last_Month": 3, "Avg_Data_Usage_GB": 120.0, "Late_Payments": 1}]}
def serve(): j = client.post("/", json=PROBE).get_json(); return j["model_version"], j["predictions"][0]["churn_probability"]
rows = []
def case(name, week, expect, **kw):
    before = serve(); S = weekly_pipeline(week, run_tag=name, **kw); after = serve()
    ok_prod = before == after
    rows.append({"case": name, "week": week, "expected": expect, "status": S["status"], "branch": S.get("branch"), "error": S.get("error"),
                 "promoted": S["promoted"], "escalated": S["escalated"], "seconds": S["seconds"], "endpoint_before": before, "endpoint_after": after,
                 "production_unchanged": ok_prod, "gate": S.get("gate")})
    print(f"### {name}: status={S['status']} promoted={S['promoted']} escalated={S['escalated']} endpoint {before}->{after} unchanged={ok_prod} t={S['seconds']}s err={S.get('error')}")
    return S

for w in range(1, 6): weekly_pipeline(w, run_tag="warmup")        # normal weeks 1-5 (week 5 challenger is rejected -> prod stays v1)
case("F1 corrupt batch (column dropped)", 6, "FAILED at validate, no retry, prod unchanged", fault="corrupt_batch", escalation=False)
case("F2 registry flaky (2 timeouts)", 3, "SUCCESS after retries (self-heals)", fault="registry_flaky", escalation=False)
case("F3 registry down (permanent)", 6, "FAILED after 3 retries w/ backoff, prod unchanged", fault="registry_down", escalation=False)
case("F4 worse challenger", 6, "gate REJECTS, prod unchanged", fault="worse_challenger", escalation=False)
case("F5 registry write fails at challenger registration", 6, "FAILED after retries, prod unchanged", fault="registry_down_on_write", escalation=False)
Path(STATE_F).unlink(missing_ok=True)
case("F6a worse challenger #1 (rule: 2 consecutive fails)", 6, "rejected", fault="worse_challenger", esc_psi=99)
case("F6b worse challenger #2", 7, "rejected", fault="worse_challenger", esc_psi=99)
case("F6c next drift week -> ESCALATE (no auto retrain)", 8, "ESCALATED by fail-count rule", esc_psi=99)
case("F7 look-ahead guard (week 3 run asks for week 4)", 3, "n/a", escalation=False) if False else None
try: ingest.fn(4, 3)
except LookAheadError as e: rows.append({"case": "F7 look-ahead guard", "expected": "LookAheadError", "status": "BLOCKED", "error": str(e)}); print("### F7 look-ahead blocked:", e)
(Path(__file__).resolve().parents[1] / "reports" / "fault_injection.json").write_text(json.dumps(rows, indent=2, default=str))
shutil.rmtree(T, ignore_errors=True)
