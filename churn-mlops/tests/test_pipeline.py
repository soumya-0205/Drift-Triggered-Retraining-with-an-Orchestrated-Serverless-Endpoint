"""Automated checks that the project's rules are enforced by code, not just described."""
import json, numpy as np, pandas as pd, pytest
from common import *
import pipeline_core as pc
import weekly_pipeline as wp

# ---------- no look-ahead ----------
def test_ingest_blocks_future_weeks():
    with pytest.raises(wp.LookAheadError): wp.ingest.fn(4, 3)
    assert len(wp.ingest.fn(3, 3)) == 500

def test_training_and_eval_frames_never_contain_future_or_overlapping_rows():
    h = pc.History()
    for k in range(1, 6): h.add(k, load_week(k))
    ids3 = set(load_week(4)["Customer_ID"]) | set(load_week(5)["Customer_ID"])
    X, recent = h.train_frame(3); ev = h.eval_frame(3)
    assert not (set(X["Customer_ID"]) & ids3) and not (set(ev["Customer_ID"]) & ids3)   # nothing from weeks 4-5 at week 3
    assert not (set(X["Customer_ID"]) & set(ev["Customer_ID"]))                           # eval rows never trained on
    assert set(recent["Customer_ID"]) <= set(X["Customer_ID"])

# ---------- validation ----------
def test_all_supplied_weeks_pass_validation():
    for w in range(1, 9): pc.validate_batch(load_week(w), w)

@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.drop(columns=["Total_Charges"]), "missing columns"),
    (lambda d: d.head(50), "too few rows"),
    (lambda d: d.assign(Churn=2), "Churn must be"),
    (lambda d: pd.concat([d, d.head(1)]), "duplicate"),
    (lambda d: d.assign(Monthly_Charges=-1.0), "negative"),
    (lambda d: d.assign(Total_Charges=np.nan), "missing-rate"),
])
def test_validation_rejects_bad_batches(mutate, msg):
    with pytest.raises(pc.ValidationError, match=msg): pc.validate_batch(mutate(load_week(1).copy()))

# ---------- promotion gate ----------
@pytest.fixture(scope="module")
def models():
    tr, _ = reference_split(); prod = build_pipeline().fit(tr[FEATURES], tr[TARGET])
    late = pd.concat([load_week(w) for w in (6, 7)]); ev = load_week(8)
    return prod, late, ev

def test_gate_rejects_identical_model(models):
    prod, _, ev = models; ok, g = pc.gate(prod, prod, ev); assert not ok and g["logloss_gain"] == 0

def test_gate_accepts_clearly_better_model(models):
    prod, late, ev = models; chal = build_pipeline().fit(late[FEATURES], late[TARGET]); ok, g = pc.gate(prod, chal, ev)
    assert ok and g["logloss_gain"] >= pc.GATE_MIN_LOGLOSS_GAIN

def test_gate_rejects_worse_model(models):
    prod, late, ev = models; y = np.random.RandomState(0).permutation(late[TARGET].values)
    bad = build_pipeline().fit(late[FEATURES], y); ok, g = pc.gate(prod, bad, ev); assert not ok and g['logloss_gain'] < 0   # worse challenger vs production -> rejected

# ---------- escalation / decision rule ----------
@pytest.fixture
def clean_state(tmp_path, monkeypatch):
    sf, hf = tmp_path / "s.json", tmp_path / "h.json"
    monkeypatch.setattr(wp, "STATE_F", sf); monkeypatch.setattr(wp, "HUMAN_F", hf); return sf, hf

def drift(psi, flag=None): return {"max_psi": psi, "drift": psi >= 0.25 if flag is None else flag, "top_feature": "Monthly_Charges"}

def test_no_drift_no_action(clean_state):   assert wp.decide.fn(1, drift(0.05), True, 1.0, 2)[0] == "no_action"
def test_moderate_drift_below_trigger_no_action(clean_state): assert wp.decide.fn(4, drift(0.17), True, 1.0, 2)[0] == "no_action"
def test_major_drift_retrains(clean_state): assert wp.decide.fn(5, drift(0.35), True, 1.0, 2)[0] == "retrain"

def test_severe_drift_escalates_then_pauses_until_human_decision(clean_state):
    sf, hf = clean_state
    b, _, st = wp.decide.fn(6, drift(1.54), True, 1.0, 2); assert b == "escalate" and st["paused_since"] == 6
    sf.write_text(json.dumps(st))
    assert wp.decide.fn(7, drift(1.9), True, 1.0, 2)[0] == "paused"                 # still no automatic retrain
    hf.write_text(json.dumps({"6": {"decision": "approve_retrain", "by": "t", "note": "n"}}))
    assert wp.decide.fn(7, drift(1.9), True, 1.0, 2)[0] == "retrain"               # resumes only with a recorded human decision

def test_two_failed_challengers_with_drift_escalate(clean_state):
    sf, _ = clean_state; sf.write_text(json.dumps({"paused_since": None, "failed_challengers": 2}))
    assert wp.decide.fn(7, drift(0.4), True, 1.0, 2)[0] == "escalate"
    assert wp.decide.fn(7, drift(0.05), True, 1.0, 2)[0] == "no_action"            # no drift -> no escalation by this rule

def test_escalation_can_be_disabled(clean_state): assert wp.decide.fn(6, drift(1.54), False, 1.0, 2)[0] == "retrain"

# ---------- drift statistics, reproducibility, config ----------
def test_psi_small_without_shift_large_with_shift():
    tr, _ = reference_split(); prof = build_reference_profile(tr)
    same = drift_report(prof, tr.sample(500, random_state=1)).psi.max()
    shifted = drift_report(prof, tr.sample(500, random_state=1).assign(Monthly_Charges=lambda d: d.Monthly_Charges + 30)).set_index("feature").psi["Monthly_Charges"]
    assert same < 0.10 and shifted > 0.25

def test_training_is_reproducible():
    tr, va = reference_split(); a = build_pipeline().fit(tr[FEATURES], tr[TARGET]); b = build_pipeline().fit(tr[FEATURES], tr[TARGET])
    assert np.array_equal(a.predict_proba(va[FEATURES]), b.predict_proba(va[FEATURES]))

def test_id_and_week_are_not_features():
    assert "Customer_ID" not in FEATURES and "Week" not in FEATURES and TARGET not in FEATURES

def test_config_values_loaded():
    assert CFG["drift_threshold_psi"] == 0.25 and CFG["escalation_psi"] == 1.0 and pc.COST_RETRAIN == CFG["cost_retraining_run_usd"]
