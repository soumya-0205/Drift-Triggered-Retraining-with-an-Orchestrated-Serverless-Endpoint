"""Driver: runs weeks 1..8 in order, one flow run per week. Between weeks it performs the (simulated) human step if the pipeline escalated."""
import json, sys, time
from weekly_pipeline import *
import weekly_pipeline as W

def main(drift_threshold=0.25, escalation=True):
    for f in (SUMMARY_F, ESC_F, STATE_F): f.unlink(missing_ok=True)
    HUMAN_F.parent.mkdir(exist_ok=True); HUMAN_F.write_text("{}")
    for w in range(1, 9):
        S = weekly_pipeline(w, drift_threshold=drift_threshold, escalation=escalation)
        print(f"week {w}: {S['status']:9s} branch={S.get('branch')} psi={S.get('max_psi')} retrained={S['retrained']} promoted={S['promoted']} prod v{S['prod_version_before']}->v{S['prod_version_after']}")
        if S["escalated"]:   # SIMULATED human: reviews the escalation and records a decision (audit trail). Not a model overwrite.
            d = json.loads(HUMAN_F.read_text()); d[str(w)] = {"decision": "approve_retrain", "by": "simulated on-call ML engineer",
                "note": f"reviewed week {w} PSI={S['max_psi']} on {S['top_feature']} (abrupt shift); approve gated retrain on next run",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            HUMAN_F.write_text(json.dumps(d, indent=2))
if __name__ == "__main__":
    main()
