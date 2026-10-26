"""Offline policy simulator: runs the SAME decision code as the Prefect flow, in memory.
Week N is scored by the model that was in production at the start of week N (decided with data <= N-1)."""
import sys, pandas as pd, numpy as np
from common import *
from pipeline_core import *

def simulate(policy, thr=None, esc_psi=None, esc_fails=None, human_delay=True, verbose=False):
    tr, va = reference_split()
    prod = build_pipeline().fit(tr[FEATURES], tr[TARGET]); prof = build_reference_profile(tr)
    hist, rows, fails, paused_week = History(), [], 0, None
    ver = 1
    for w in range(1, 9):
        df = load_week(w); validate_batch(df, w)
        m = evaluate(prod, df); r = {"week": w, "policy": policy, "prod_version": ver, **{k: m[k] for k in ("roc_auc","log_loss","f1","recall")}}
        hist.add(w, df)
        dr = drift_report(prof, df); score = float(dr.psi.max()); r["max_psi"] = score; r["top_feature"] = dr.sort_values("psi").feature.iloc[-1]
        retrain = {"never": False, "weekly": True, "drift": score >= (thr or 9e9)}[policy.split("|")[0]]
        r["drift"] = score >= (thr if thr else 0.10); r["escalated"] = False; r["retrained"] = r["promoted"] = False; r["logloss_gain"] = np.nan
        human_ok = paused_week is not None   # a human signed off the pause raised in an earlier week
        if esc_psi and score >= esc_psi and not human_ok:
            r["escalated"] = True; paused_week = w; retrain = False      # STOP: no automatic retrain/promotion
        elif esc_fails and fails >= esc_fails and not human_ok:
            r["escalated"] = True; paused_week = w; retrain = False
        if paused_week is not None and not r["escalated"] and w > paused_week:
            retrain = True; paused_week = None; fails = 0                # human-approved retrain, still through the gate
        r["cost_usd"] = business_cost(prod, df)
        if retrain:
            chal, newprof, n = train_challenger(hist, w); ok, g = gate(prod, chal, hist.eval_frame(w))
            r["retrained"] = True; r["logloss_gain"] = g["logloss_gain"]; r["cost_usd"] += COST_RETRAIN
            if ok: prod, prof, ver, r["promoted"], fails = chal, newprof, ver + 1, True, 0
            else: fails += 1
        rows.append(r)
    return pd.DataFrame(rows)

POLICIES = {
    "never":                        dict(policy="never"),
    "weekly":                       dict(policy="weekly"),
    "drift PSI>=0.10":              dict(policy="drift", thr=0.10),
    "drift PSI>=0.25":              dict(policy="drift", thr=0.25),
    "drift PSI>=0.25 + escalation": dict(policy="drift", thr=0.25, esc_psi=1.0, esc_fails=2),
}
if __name__ == "__main__":
    out = []
    for name, kw in POLICIES.items():
        d = simulate(**kw); d["policy"] = name; out.append(d)
        print(f"\n== {name}: retrains={d.retrained.sum()} promos={d.promoted.sum()} cost=${d.cost_usd.sum():,.0f} meanAUC={d.roc_auc.mean():.3f} meanLL={d.log_loss.mean():.3f}")
        print(d[["week","prod_version","roc_auc","log_loss","max_psi","drift","escalated","retrained","promoted","logloss_gain","cost_usd"]].round(3).to_string(index=False))
    pd.concat(out).to_csv("../reports/policy_weekly.csv", index=False)
