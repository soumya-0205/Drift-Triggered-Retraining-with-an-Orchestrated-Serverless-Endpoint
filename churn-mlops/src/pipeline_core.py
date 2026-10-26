"""Pure decision logic shared by the Prefect flow and the offline policy simulator.
No registry or orchestrator imports here, so the same code is used in both places."""
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from common import (CFG, reference_split, FEATURES, TARGET, NUM, CAT, REQUIRED_COLS, SEED, build_pipeline, evaluate,
                    build_reference_profile, drift_report, load_reference, load_week, THRESHOLD)

# ---------- configuration (documented in README) ----------
WINDOW_WEEKS = CFG["window_weeks"]                    # challenger trains on the last N weeks of "train halves"
REF_WEIGHT = CFG["reference_weight"]                  # old reference rows are down-weighted vs. recent weekly rows
EVAL_HALF = CFG["eval_half_fraction"]                 # share of each weekly batch held out as a never-trained-on "eval half"
GATE_MIN_LOGLOSS_GAIN = CFG["gate_min_logloss_gain"]  # challenger must cut log loss by >= 2% ...
GATE_MAX_AUC_DROP = CFG["gate_max_auc_drop"]          # ... and not lose more than 0.01 ROC-AUC
COST_FN, COST_FP, COST_RETRAIN = CFG["cost_missed_churner_usd"], CFG["cost_false_alert_usd"], CFG["cost_retraining_run_usd"]

class ValidationError(Exception):
    pass

def split_week(df, w):
    """Fixed split of every weekly batch: 60% train half (A), 40% eval half (B)."""
    a, b = train_test_split(df, test_size=EVAL_HALF, stratify=df[TARGET], random_state=SEED + w)
    return a, b

def validate_batch(df, w=None):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: raise ValidationError(f"missing columns: {missing}")
    if len(df) < 100: raise ValidationError(f"too few rows: {len(df)}")
    if not set(df[TARGET].dropna().unique()) <= {0, 1} or df[TARGET].isna().any():
        raise ValidationError("Churn must be 0/1 with no nulls")
    if df["Customer_ID"].duplicated().any(): raise ValidationError("duplicate Customer_ID")
    bad = [c for c in NUM if df[c].isna().mean() > 0.10]
    if bad: raise ValidationError(f"missing-rate > 10% in {bad}")
    if (df[NUM].dropna() < 0).any().any(): raise ValidationError("negative values in numeric features")
    unseen = {c: sorted(set(df[c].dropna()) - set(load_reference()[c].unique())) for c in CAT}
    return {"rows": len(df), "unseen_categories": {k: v for k, v in unseen.items() if v}}

class History:
    """Per-week labelled data visible so far (enforces no look-ahead by construction)."""
    def __init__(self): self.A, self.B = {}, {}
    def add(self, w, df):
        self.A[w], self.B[w] = split_week(df, w)
    def train_frame(self, upto):
        ref_tr, _ = reference_split()
        parts = [self.A[k] for k in sorted(self.A) if upto - WINDOW_WEEKS < k <= upto]
        X = pd.concat([ref_tr.assign(_w=REF_WEIGHT)] + [p.assign(_w=1.0) for p in parts], ignore_index=True)
        return X, pd.concat(parts, ignore_index=True)
    def eval_frame(self, upto):
        ks = [k for k in sorted(self.B) if upto - 1 <= k <= upto]   # eval halves of last 2 weeks
        return pd.concat([self.B[k] for k in ks], ignore_index=True)

def train_challenger(hist, upto):
    X, recent = hist.train_frame(upto)
    m = build_pipeline().fit(X[FEATURES], X[TARGET], clf__sample_weight=X["_w"].values)
    return m, build_reference_profile(recent), len(X)

def gate(prod, chal, ev):
    p, c = evaluate(prod, ev), evaluate(chal, ev)
    gain = (p["log_loss"] - c["log_loss"]) / p["log_loss"]
    ok = gain >= GATE_MIN_LOGLOSS_GAIN and c["roc_auc"] >= p["roc_auc"] - GATE_MAX_AUC_DROP
    return ok, {"prod": p, "chal": c, "logloss_gain": float(gain)}

def business_cost(model, df, retrains=0):
    p = model.predict_proba(df[FEATURES])[:, 1]; yh = p >= THRESHOLD; y = df[TARGET].values
    return float(((~yh) & (y == 1)).sum() * COST_FN + (yh & (y == 0)).sum() * COST_FP + retrains * COST_RETRAIN)
