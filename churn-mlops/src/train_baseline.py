"""Task 1: train baseline on the reference data, evaluate on a held-out slice, register + promote v1."""
import numpy as np, random
from common import *
import registry

def main():
    np.random.seed(SEED); random.seed(SEED)
    tr, va = reference_split()
    model = build_pipeline().fit(tr[FEATURES], tr[TARGET])
    metrics = {f"holdout_{k}": v for k, v in evaluate(model, va).items()}
    metrics.update({f"train_{k}": v for k, v in evaluate(model, tr).items() if k in ("roc_auc", "pr_auc")})
    profile = build_reference_profile(tr)
    v = registry.register(model, profile, metrics,
        params={"model": "LogisticRegression(C=1)", "threshold": THRESHOLD, "seed": SEED, "train_rows": len(tr), "holdout_rows": len(va)},
        tags={"role": "baseline", "trained_on": "churn_reference_train.csv (75% split)"}, run_name="baseline")
    registry.promote(v)
    save_json(metrics, ROOT / "reports" / "baseline_metrics.json")
    print("registered + promoted version", v); print({k: round(x, 3) for k, x in metrics.items()})

if __name__ == "__main__":
    main()
