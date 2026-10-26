"""Cross-check: my own PSI/KS implementation vs Evidently (open-source drift library) on every week.
Reference = the baseline training split (same data as the stored reference profile)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from common import *

tr, _ = reference_split(); prof = build_reference_profile(tr)
dd = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
ref = Dataset.from_pandas(tr[FEATURES], data_definition=dd)
rows = []
for w in range(1, 9):
    df = load_week(w); cur = Dataset.from_pandas(df[FEATURES], data_definition=dd)
    mine = drift_report(prof, df).set_index("feature")
    ev_psi = Report([DataDriftPreset(method="psi")]).run(cur, ref)
    ev_def = Report([DataDriftPreset()]).run(cur, ref)       # Evidently defaults: KS (numeric) / chi2 (categorical), p<0.05
    if w == 6: ev_def.save_html(str(ROOT / "reports" / "evidently_week6_drift_report.html"))
    pv = {m["metric_name"].split("column=")[1].split(",")[0]: m["value"] for m in ev_psi.dict()["metrics"] if m["metric_name"].startswith("ValueDrift")}
    dv = {m["metric_name"].split("column=")[1].split(",")[0]: m["value"] for m in ev_def.dict()["metrics"] if m["metric_name"].startswith("ValueDrift")}
    for c in FEATURES:
        rows.append({"week": w, "feature": c, "my_psi": mine.loc[c, "psi"], "evidently_psi": pv[c], "my_ks_p": mine.loc[c, "ks_p"], "evidently_default_stat": dv[c]})
D = pd.DataFrame(rows); D.to_csv(ROOT / "reports" / "evidently_crosscheck.csv", index=False)
rho = spearmanr(D.my_psi, D.evidently_psi).correlation
wk = D.groupby("week").agg(my_max=("my_psi", "max"), ev_max=("evidently_psi", "max"))
wk["mine_flag_0.25"] = wk.my_max >= 0.25; wk["evidently_flag_0.25"] = wk.ev_max >= 0.25
wk["top_feature_mine"] = D.loc[D.groupby("week").my_psi.idxmax()].set_index("week").feature
wk["top_feature_evidently"] = D.loc[D.groupby("week").evidently_psi.idxmax()].set_index("week").feature
wk.round(3).to_csv(ROOT / "reports" / "evidently_weekly_max_psi.csv")
out = {"spearman_rank_correlation_all_feature_weeks": round(float(rho), 3), "weeks_with_same_0.25_flag": int((wk["mine_flag_0.25"] == wk["evidently_flag_0.25"]).sum()),
       "weeks_with_same_top_feature": int((wk.top_feature_mine == wk.top_feature_evidently).sum()), "n_feature_weeks": len(D)}
save_json(out, ROOT / "reports" / "evidently_summary.json"); print(wk.round(3).to_string()); print(out)
