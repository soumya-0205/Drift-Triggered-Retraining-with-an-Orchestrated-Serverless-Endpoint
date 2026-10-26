import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT
d = pd.read_csv(ROOT / "reports" / "policy_weekly.csv")
S = d.groupby("policy", sort=False).agg(retrains=("retrained", "sum"), promotions=("promoted", "sum"), mean_auc=("roc_auc", "mean"),
      mean_logloss=("log_loss", "mean"), logloss_w6_8=("log_loss", lambda s: s.iloc[5:].mean()), total_cost_usd=("cost_usd", "sum")).round(3)
S["cost_vs_never_usd"] = (S.total_cost_usd - S.loc["never", "total_cost_usd"]).round(0)
S.to_csv(ROOT / "reports" / "policy_summary.csv"); print(S.to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for p, g in d.groupby("policy", sort=False):
    ax[0].plot(g.week, g.log_loss, marker="o", ms=3, label=p, lw=1.6); ax[1].plot(g.week, g.cost_usd.cumsum() / 1000, marker="o", ms=3, lw=1.6)
ax[0].set_title("log loss of the model serving each week"); ax[1].set_title("cumulative cost (USD thousands)")
for a in ax: a.set_xlabel("week"); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=6.5); plt.tight_layout(); plt.savefig(ROOT / "reports" / "fig_policy_comparison.png", dpi=130)
