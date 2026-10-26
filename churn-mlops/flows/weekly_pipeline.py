"""Task 4 / 6 / 7: Prefect orchestration (open-source stand-in for AWS Step Functions).

Stages: ingest -> validate -> load production -> score deployed model -> drift check ->
        DECIDE -> {retrain -> compare -> promote | skip} | {escalate (STOP)} | {no action} -> record summary
Every branch (taken or skipped) is a named task, so the run graph shows what happened and what was skipped."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[1] / ".prefect"))
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")
import numpy as np, pandas as pd
from prefect import flow, task, get_run_logger
from prefect.runtime import task_run
from prefect.artifacts import create_markdown_artifact
from common import *
import registry
from pipeline_core import *

STATE_DIR = Path(os.environ.get("PIPELINE_STATE_DIR", ROOT / "state")); STATE_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR = Path(os.environ.get("PIPELINE_REPORT_DIR", ROOT / "reports")); REPORT_DIR.mkdir(exist_ok=True, parents=True)
STATE_F, SUMMARY_F, ESC_F = STATE_DIR / "pipeline_state.json", REPORT_DIR / "run_summaries.jsonl", REPORT_DIR / "escalations.jsonl"
HUMAN_F = Path(os.environ.get("HUMAN_DECISIONS", ROOT / "configs" / "human_decisions.json"))

class LookAheadError(Exception): ...

def _only_transient(task, task_run, state):
    """Retry only infrastructure-type errors. Deterministic errors (bad data, look-ahead) are never retried."""
    exc = state.result(raise_on_failure=False)
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))

def read_state():
    return json.loads(STATE_F.read_text()) if STATE_F.exists() else {"paused_since": None, "failed_challengers": 0}

# ---------------------------------------------------------------- tasks
@task(name="1-ingest", retries=2, retry_delay_seconds=2, retry_condition_fn=_only_transient)   # file/network reads are transient
def ingest(week: int, current_week: int, fault=None):
    if week > current_week: raise LookAheadError(f"week {week} requested during week {current_week}")
    df = load_week(week)
    if fault == "corrupt_batch": df = df.drop(columns=["Total_Charges"])        # injected fault
    return df

@task(name="2-validate", retries=0)   # deliberately NO retry: a corrupt batch will not fix itself; fail fast and report
def validate(df, week):
    return validate_batch(df, week)

@task(name="3-load-production", retries=3, retry_delay_seconds=[1, 2, 4], retry_condition_fn=_only_transient)  # exponential-style backoff for registry blips
def load_production(fault=None):
    attempt = task_run.run_count if task_run.run_count else 1
    if fault == "registry_down": raise ConnectionError("registry unavailable (injected, permanent)")
    if fault == "registry_flaky" and attempt < 3: raise ConnectionError(f"registry timeout (injected, attempt {attempt})")
    return registry.load_production()

@task(name="4-score-deployed-model")
def score(model, df): return evaluate(model, df)

@task(name="5-drift-check")
def drift_check(profile, df, thr):
    rep = drift_report(profile, df); top = rep.sort_values("psi").iloc[-1]
    return {"max_psi": float(top.psi), "top_feature": top.feature, "drift": bool(top.psi >= thr),
            "per_feature_psi": rep.set_index("feature").psi.round(4).to_dict()}

@task(name="6-decide")
def decide(week, drift, esc_enabled, esc_psi, esc_fails):
    """Branch selector. Escalation is checked BEFORE retraining so automation can never override it."""
    st, why = read_state(), []
    if st["paused_since"] is not None:
        dec = json.loads(HUMAN_F.read_text()).get(str(st["paused_since"])) if HUMAN_F.exists() else None
        if dec is None: return "paused", ["pipeline paused since week %s awaiting human decision" % st["paused_since"]], st
        st["paused_since"] = None
        if dec["decision"] == "approve_retrain": st["failed_challengers"] = 0; return "retrain", [f"human-approved retrain ({dec['by']}: {dec['note']})"], st
        return "no_action", ["human rejected retraining"], st
    if esc_enabled and drift["max_psi"] >= esc_psi:
        st["paused_since"] = week; return "escalate", [f"max PSI {drift['max_psi']:.2f} >= {esc_psi} on {drift['top_feature']}"], st
    if esc_enabled and drift["drift"] and st["failed_challengers"] >= esc_fails:
        st["paused_since"] = week; return "escalate", [f"{st['failed_challengers']} consecutive challengers failed the gate"], st
    if drift["drift"]: return "retrain", [f"drift: max PSI {drift['max_psi']:.2f} on {drift['top_feature']}"], st
    return "no_action", ["no drift"], st

@task(name="7a-retrain")
def retrain(week, fault=None):
    hist = History()
    for k in range(1, week + 1): hist.add(k, load_week(k))      # only weeks 1..N are ever read
    if fault == "worse_challenger":                                # injected: train on shuffled labels
        X, recent = hist.train_frame(week); X = X.copy(); X[TARGET] = np.random.RandomState(0).permutation(X[TARGET].values)
        m = build_pipeline().fit(X[FEATURES], X[TARGET], clf__sample_weight=X["_w"].values); prof, n = build_reference_profile(recent), len(X)
    else:
        m, prof, n = train_challenger(hist, week)
    return m, prof, n, hist.eval_frame(week)

@task(name="7b-compare-on-recent-data")
def compare(prod_model, chal, ev):
    ok, g = gate(prod_model, chal, ev); return ok, g

@task(name="7c-register-challenger", retries=3, retry_delay_seconds=[1, 2, 4], retry_condition_fn=_only_transient)
def register_challenger(chal, prof, g, week, prod_version, fault=None):
    if fault == "registry_down_on_write": raise ConnectionError("registry write failed (injected, permanent)")
    return registry.register(chal, prof, {**{f"chal_{k}": v for k, v in g["chal"].items()}, "logloss_gain_vs_prod": g["logloss_gain"]},
                             params={"week": week, "window_weeks": WINDOW_WEEKS, "ref_weight": REF_WEIGHT},
                             tags={"role": "challenger", "week": week, "prod_at_training": prod_version}, run_name=f"challenger_w{week}")

@task(name="8a-promote", retries=3, retry_delay_seconds=[1, 2, 4], retry_condition_fn=_only_transient)
def promote(version): return registry.promote(version)

@task(name="8b-skip-promotion")
def skip_promotion(reason): return {"skipped": True, "reason": reason}

@task(name="7x-skip-retrain")
def skip_retrain(reason): return {"skipped": True, "reason": reason}

@task(name="9-escalate-to-human")
def escalate(week, reasons, drift, scored):
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "week": week, "reasons": reasons, "max_psi": drift["max_psi"],
           "per_feature_psi": drift["per_feature_psi"], "deployed_model_metrics": {k: round(v, 3) for k, v in scored.items()},
           "action": "PIPELINE PAUSED - no automatic retrain or promotion until a human decision is recorded in configs/human_decisions.json"}
    with open(ESC_F, "a") as f: f.write(json.dumps(rec) + "\n")
    get_run_logger().critical("ESCALATION_TRIGGERED week=%s reasons=%s", week, reasons)
    create_markdown_artifact(key="escalation", markdown=f"## ESCALATION - week {week}\n- reasons: {reasons}\n- max PSI: {drift['max_psi']:.2f}\n- {rec['action']}")
    return rec

@task(name="10-record-summary")
def record_summary(summary):
    with open(SUMMARY_F, "a") as f: f.write(json.dumps(summary, default=str) + "\n")
    return summary

# ---------------------------------------------------------------- flow
@flow(name="weekly-churn-pipeline", retries=0)
def weekly_pipeline(week: int, drift_threshold: float = CFG["drift_threshold_psi"], escalation: bool = CFG["escalation_enabled"],
                    esc_psi: float = CFG["escalation_psi"], esc_fails: int = CFG["escalation_consecutive_failed_challengers"], fault: str | None = None, run_tag: str = "main"):
    log = get_run_logger(); t0 = time.time()
    S = {"run_tag": run_tag, "week": week, "fault": fault, "status": "RUNNING", "drift_detected": None, "retrained": False, "promoted": False,
         "escalated": False, "prod_version_before": None, "prod_version_after": None, "data_weeks_used": f"1..{week}"}
    try:
        df = ingest(week, week, fault); S["rows"] = len(df)
        validate(df, week)
        ver, model, prof = load_production(fault); S["prod_version_before"] = ver
        sc = score(model, df); S["deployed_model_metrics"] = {k: round(v, 4) for k, v in sc.items()}
        dr = drift_check(prof, df, drift_threshold); S.update(drift_detected=dr["drift"], max_psi=round(dr["max_psi"], 4), top_feature=dr["top_feature"])
        branch, reasons, st = decide(week, dr, escalation, esc_psi, esc_fails); S["branch"], S["reasons"] = branch, reasons
        if branch == "escalate":
            escalate(week, reasons, dr, sc); S["escalated"] = True
            skip_retrain("escalated to human"); skip_promotion("escalated to human")
        elif branch == "retrain":
            chal, cprof, n, ev = retrain(week, fault); S["retrained"] = True
            ok, g = compare(model, chal, ev); S["gate"] = {"passed": ok, "logloss_gain": round(g["logloss_gain"], 4),
                "prod": {k: round(g["prod"][k], 4) for k in ("roc_auc", "log_loss")}, "chal": {k: round(g["chal"][k], 4) for k in ("roc_auc", "log_loss")}}
            cv = register_challenger(chal, cprof, g, week, ver, fault); S["challenger_version"] = cv
            if ok: promote(cv); S["promoted"] = True; st["failed_challengers"] = 0
            else: skip_promotion(f"challenger failed gate (log-loss gain {g['logloss_gain']:.3f})"); st["failed_challengers"] += 1
        else:
            skip_retrain(reasons[0]); skip_promotion("no challenger")
        STATE_F.write_text(json.dumps(st))
        S["prod_version_after"] = registry.production_version(); S["status"] = "ESCALATED" if S["escalated"] else "SUCCESS"
    except Exception as e:                                         # any failure: report clearly, production untouched
        S["status"] = "FAILED"; S["error"] = f"{type(e).__name__}: {e}"
        try: S["prod_version_after"] = registry.production_version()
        except Exception: S["prod_version_after"] = "unknown (registry unreachable)"
        log.error("RUN FAILED week=%s: %s | production model unchanged (version %s)", week, S["error"], S["prod_version_after"])
    S["seconds"] = round(time.time() - t0, 2)
    record_summary(S); return S
