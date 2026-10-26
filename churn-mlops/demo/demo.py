"""Guided live demo (about 5 minutes). Works on Windows/macOS/Linux. Uses its OWN registry in ./demo_run, so it never touches the evidence files.

    python demo/demo.py            # interactive: pauses between steps, you play the human on-call at the escalation
    python demo/demo.py --auto     # no pauses (rehearsal / fallback)

Optional (to show the orchestrator UI): in another terminal run `prefect server start`, then in THIS terminal set
PREFECT_API_URL=http://127.0.0.1:4200/api before starting the demo; the 8 runs appear at http://127.0.0.1:4200/runs."""
import os, sys, json, time, shutil, socket, subprocess, argparse, urllib.request, urllib.error, webbrowser
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / "demo_run"
ap = argparse.ArgumentParser(); ap.add_argument("--auto", action="store_true"); AUTO = ap.parse_args().auto
shutil.rmtree(RUN, ignore_errors=True); RUN.mkdir()
os.environ.update(MLFLOW_TRACKING_URI=f"sqlite:///{RUN.as_posix()}/mlflow.db", ARTIFACT_ROOT=str(RUN / "mlartifacts"), PIPELINE_STATE_DIR=str(RUN / "state"),
                  PIPELINE_REPORT_DIR=str(RUN / "reports"), HUMAN_DECISIONS=str(RUN / "human_decisions.json"), REFRESH_SECONDS="0.5",
                  PREFECT_HOME=str(RUN / "prefect_home"), PREFECT_LOGGING_LEVEL="CRITICAL", PREFECT_SERVER_LOGGING_LEVEL="CRITICAL", MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR="false",
                  TQDM_DISABLE="1", PYTHONWARNINGS="ignore", MLFLOW_LOGGING_LEVEL="ERROR")
(RUN / "human_decisions.json").write_text("{}")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "flows")]

def banner(t): print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)
def pause(msg="Press Enter to continue..."):
    if not AUTO: input(f"\n  [{msg}] ")
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

banner("STEP 1 - Baseline model trained on the reference data and registered as production")
import logging
for _n in ("mlflow", "docket", "alembic"): logging.getLogger(_n).setLevel(logging.CRITICAL)
import train_baseline, registry
train_baseline.main()
print("\n  Registry now holds:", [(v.version, v.tags.get("role")) for v in registry._setup().search_model_versions("name='churn-model'")],
      "| alias production -> v%d" % registry.production_version())
print("  Stored with the model: parameters, evaluation metrics and the reference profile (bin edges, category frequencies, missing rates, KS sample).")
pause()

banner("STEP 2 - Serverless-style endpoint (functions-framework) serving whatever the registry alias 'production' points to")
PORT = free_port(); URL = f"http://127.0.0.1:{PORT}/"
srv = subprocess.Popen([sys.executable, "-m", "functions_framework", "--source", str(ROOT / "function" / "main.py"), "--target", "predict", "--port", str(PORT)],
                       env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PROBE = {"instances": [{"Tenure_Months": 5, "Contract_Type": "Month-to-month", "Internet_Service": "Fiber", "Monthly_Charges": 85.0,
                        "Total_Charges": 400.0, "Support_Calls_Last_Month": 3, "Avg_Data_Usage_GB": 120.0, "Late_Payments": 1}]}
def call(payload=PROBE):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try: return json.loads(urllib.request.urlopen(req, timeout=15).read())
    except urllib.error.HTTPError as e: return {"HTTP": e.code, "body": json.loads(e.read())}
for _ in range(60):
    try: r = call(); break
    except Exception: time.sleep(0.5)
else: sys.exit("endpoint did not start")
print("  POST a customer ->", json.dumps(r))
print("  Bad request (missing features) ->", json.dumps(call({"instances": [{"Tenure_Months": 5}]})))
print("  (Cold start ~2.7 s vs warm ~6 ms in our latency study: reports/latency.json)")
pause()

banner("STEP 3 - Eight weeks through the orchestrated workflow (Prefect). Each week: ingest > validate > drift > decide > [retrain > compare > promote]")
from weekly_pipeline import weekly_pipeline, HUMAN_F
def probe_line(S):
    r = call(); return f"endpoint serves v{r['model_version']} (p={r['predictions'][0]['churn_probability']})"
for w in range(1, 9):
    if w in (5, 6, 7): pause(f"Next: week {w}")
    S = weekly_pipeline(w)
    time.sleep(0.8)
    line = f"  week {w}: max PSI {S['max_psi']:<6} ({S['top_feature']:<15}) -> {S['branch']:<10}"
    if S["retrained"]: line += f" | challenger log-loss gain {S['gate']['logloss_gain']*100:+.1f}% -> " + ("PROMOTED" if S["promoted"] else "REJECTED by gate")
    print(line + f" | {probe_line(S)}")
    if S["escalated"]:
        print("\n  " + "!" * 70 + "\n  ESCALATION: severe drift (max PSI >= 1.0). Pipeline PAUSED - no automatic retrain or promotion.\n  " + "!" * 70)
        print("  Record written to demo_run/reports/escalations.jsonl; the Prefect run shows the 'escalation' artifact.")
        who = "demo presenter (on-call engineer)"
        if not AUTO:
            ans = input("\n  You are the human on call. Type 'approve' to allow a GATED retrain next week (anything else rejects): ").strip().lower()
            dec = "approve_retrain" if ans == "approve" else "reject"
        else: dec = "approve_retrain"
        d = json.loads(HUMAN_F.read_text()); d[str(w)] = {"decision": dec, "by": who, "note": f"reviewed week {w} PSI={S['max_psi']}", "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        HUMAN_F.write_text(json.dumps(d, indent=2)); print(f"  Decision recorded: {dec}")
print(f"\n  >>> Production went v1 -> v{registry.production_version()} WITHOUT touching or restarting the endpoint code. Promotion only happens inside the workflow, after the gate.")
pause()

banner("STEP 4 - Failure handling: production and endpoint must not change")
before = (registry.production_version(), call()["predictions"][0]["churn_probability"])
for label, kw in (("corrupt batch (column dropped)", dict(fault="corrupt_batch", escalation=False)),
                  ("worse challenger (forced retrain with shuffled labels)", dict(fault="worse_challenger", drift_threshold=0.0, escalation=False))):
    S = weekly_pipeline(8, run_tag="demo-fault", **kw); after = (registry.production_version(), call()["predictions"][0]["churn_probability"])
    print(f"  {label}\n     run status: {S['status']} | promoted: {S['promoted']} | error: {S.get('error')} | gate: {S.get('gate',{}).get('passed')}\n     production/endpoint before {before} after {after} -> unchanged: {before == after}")
pause()

banner("STEP 5 - Which retraining policy? (8-week simulation, same decision code)")
import pandas as pd
T = pd.read_csv(ROOT / "reports" / "policy_summary.csv")[["policy", "retrains", "promotions", "mean_logloss", "logloss_w6_8", "total_cost_usd"]]
print(T.to_string(index=False)); print("\n  Recommendation: drift-triggered at PSI >= 0.25 + promotion gate, with escalation at PSI >= 1.0 as the safety layer.")
for f in ("fig_policy_timeline.png", "fig_psi_heatmap.png"):
    p = ROOT / "reports" / f
    if not AUTO: webbrowser.open(p.as_uri())
srv.terminate()
banner("DONE - evidence: evidence/ (Prefect + MLflow screenshots), reports/REPORT.pdf")
