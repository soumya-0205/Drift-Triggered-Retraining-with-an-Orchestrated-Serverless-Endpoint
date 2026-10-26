"""Shared helpers: data loading, model pipeline, metrics, drift statistics."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             recall_score, precision_score, log_loss, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
CFG = json.loads((ROOT / "configs" / "pipeline_config.json").read_text())
NUM = ["Tenure_Months", "Monthly_Charges", "Total_Charges",
       "Support_Calls_Last_Month", "Avg_Data_Usage_GB", "Late_Payments"]
CAT = ["Contract_Type", "Internet_Service"]
FEATURES = NUM + CAT
TARGET = "Churn"
REQUIRED_COLS = ["Customer_ID", *FEATURES, TARGET]
THRESHOLD = CFG["decision_threshold"]  # decision threshold for the ~20% positive rate (recall-leaning)


def load_reference():
    return pd.read_csv(DATA / "churn_reference_train.csv")


def load_week(w):
    return pd.read_csv(DATA / f"churn_week_{w}.csv")


def reference_split():
    """Fixed 75/25 stratified split of the reference file: train for the baseline, hold-out for its evaluation."""
    from sklearn.model_selection import train_test_split
    ref = load_reference()
    return train_test_split(ref, test_size=0.25, stratify=ref[TARGET], random_state=SEED)


def build_pipeline(seed=SEED):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model, df, thr=THRESHOLD):
    p = model.predict_proba(df[FEATURES])[:, 1]
    y = df[TARGET].values
    yh = (p >= thr).astype(int)
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, yh)),
            "recall": float(recall_score(y, yh)),
            "precision": float(precision_score(y, yh, zero_division=0)),
            "log_loss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "mean_pred": float(p.mean()), "positive_rate": float(y.mean())}


# ---------------- drift statistics ----------------
def psi_numeric(ref, cur, edges):
    ref, cur = np.asarray(ref.dropna()), np.asarray(cur.dropna())
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_cat(ref_freq, cur, cats):
    r = pd.Series(ref_freq).reindex(cats).fillna(0).clip(lower=1e-4).values
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4).values
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference_profile(ref):
    """Everything a later drift check needs: bin edges, moments, category
    frequencies, missing rates and a stored sample for KS tests."""
    prof = {"n_rows": len(ref), "positive_rate": float(ref[TARGET].mean()),
            "numeric": {}, "categorical": {}}
    for c in NUM:
        edges = np.unique(np.quantile(ref[c].dropna(), np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        prof["numeric"][c] = {
            "mean": float(ref[c].mean()), "std": float(ref[c].std()),
            "missing_rate": float(ref[c].isna().mean()),
            "edges": [None if np.isinf(e) else float(e) for e in edges],
            "sample": ref[c].dropna().sample(min(1500, int(ref[c].notna().sum())),
                                             random_state=SEED).round(4).tolist()}
    for c in CAT:
        prof["categorical"][c] = ref[c].value_counts(normalize=True).round(6).to_dict()
    return prof


def _edges(e):
    out = list(e)
    out[0], out[-1] = -np.inf, np.inf
    return np.array(out, dtype=float)


def drift_report(prof, cur):
    """Per-feature PSI + KS (numeric) / PSI (categorical) vs the stored reference profile."""
    rows = []
    for c in NUM:
        p = prof["numeric"][c]
        s = pd.Series(p["sample"])
        ks = ks_2samp(s, cur[c].dropna())
        rows.append({"feature": c, "psi": psi_numeric(s, cur[c], _edges(p["edges"])),
                     "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)})
    for c in CAT:
        cats = list(prof["categorical"][c].keys())
        rows.append({"feature": c, "psi": psi_cat(prof["categorical"][c], cur[c], cats),
                     "ks_stat": np.nan, "ks_p": np.nan})
    return pd.DataFrame(rows)


def save_json(o, p):
    Path(p).write_text(json.dumps(o, indent=2, default=str))
