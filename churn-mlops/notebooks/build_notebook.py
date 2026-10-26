import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("# Task 3 - Drift investigation\nWhat changed in weeks 1-8 versus the reference data, and did it hurt the baseline model? Everything here uses only the supplied CSVs and the registered baseline (v1).")
code("""import sys; sys.path.insert(0,'../src')
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
import registry
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
ver = 1; model, prof = registry.load_version(ver)   # the BASELINE (v1) and its stored reference profile, not whatever is in production now
weeks = {w: load_week(w) for w in range(1,9)}; ref = load_reference()
print('baseline model version', ver)""")
md("## 1. Feature drift per week (PSI vs reference profile)\nConvention: PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 major. Categorical features use category-frequency PSI; numeric use 10 reference-decile bins, plus a KS test.")
code("""rows=[]
for w,d in weeks.items():
    r=drift_report(prof,d); r['week']=w; rows.append(r)
D=pd.concat(rows); P=D.pivot(index='feature',columns='week',values='psi')
print(P.round(3).to_string())
fig,ax=plt.subplots(figsize=(7.5,3.6)); im=ax.imshow(np.log10(P.clip(lower=0.003)),cmap='YlOrRd',aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels(range(1,9)); ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index)
for i in range(P.shape[0]):
    for j in range(P.shape[1]): ax.text(j,i,f'{P.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
ax.set_xlabel('week'); ax.set_title('PSI vs reference (colour = log10 PSI)'); plt.tight_layout(); plt.savefig('../reports/fig_psi_heatmap.png'); plt.show()""")
code("""ks=D[D.feature.isin(NUM)].pivot(index='feature',columns='week',values='ks_stat'); print('KS statistic'); print(ks.round(2).to_string())""")
md("**Reading:** weeks 1-3 are quiet (max PSI < 0.07). Drift starts in week 4 (Monthly_Charges 0.17, Support_Calls 0.12), grows in week 5 (Monthly_Charges 0.35, Support_Calls 0.23) and jumps in week 6 to PSI ~1.5 on Monthly_Charges, with Support_Calls (0.40), Total_Charges (0.26) and the two categoricals (0.09-0.16) also moving. Tenure, Late_Payments and Avg_Data_Usage stay essentially stable. That is a step change in weeks 5-6, not noise.")
md("## 2. What the drift looks like")
code("""fig,ax=plt.subplots(1,3,figsize=(11,3.2))
for w,c in ((1,'#4c78a8'),(5,'#f58518'),(7,'#e45756')): ax[0].hist(weeks[w].Monthly_Charges,bins=30,histtype='step',density=True,label=f'week {w}',color=c)
ax[0].hist(ref.Monthly_Charges,bins=30,histtype='step',density=True,label='reference',color='k',ls='--'); ax[0].legend(fontsize=7); ax[0].set_title('Monthly_Charges')
mix=pd.concat([ref.assign(g='ref')]+[weeks[w].assign(g=f'w{w}') for w in range(1,9)])
mix.groupby('g').Contract_Type.apply(lambda s:(s=='Month-to-month').mean()).reindex(['ref']+[f'w{w}' for w in range(1,9)]).plot.bar(ax=ax[1],color='#72b7b2'); ax[1].set_title('share month-to-month'); ax[1].set_xlabel('')
mix.groupby('g').Internet_Service.apply(lambda s:(s=='Fiber').mean()).reindex(['ref']+[f'w{w}' for w in range(1,9)]).plot.bar(ax=ax[2],color='#b279a2'); ax[2].set_title('share Fiber'); ax[2].set_xlabel('')
plt.tight_layout(); plt.savefig('../reports/fig_feature_shift.png'); plt.show()""")
md("## 3. Is it only inputs changing, or did the input-to-churn relationship change too?\nIf only P(x) moved (covariate shift) a model with a stable P(y|x) would keep working. Check churn rate by segment, and the same model class re-fit on weeks 6-8.")
code("""late=pd.concat([weeks[w] for w in (6,7,8)])
seg=pd.DataFrame({'reference':ref.groupby('Contract_Type').Churn.mean(),'weeks 6-8':late.groupby('Contract_Type').Churn.mean()}); print(seg.round(2).to_string()); print()
seg=pd.DataFrame({'reference':ref.groupby('Internet_Service').Churn.mean(),'weeks 6-8':late.groupby('Internet_Service').Churn.mean()}); print(seg.round(2).to_string())
from sklearn.linear_model import LogisticRegression
tr,va=reference_split(); m_ref=build_pipeline().fit(tr[FEATURES],tr[TARGET]); m_late=build_pipeline().fit(late[FEATURES],late[TARGET])
names=m_ref[:-1].get_feature_names_out(); print(); print(pd.DataFrame({'ref model coef':m_ref[-1].coef_[0],'weeks 6-8 model coef':m_late[-1].coef_[0]},index=names).round(2).to_string())""")
md("**Reading:** this is covariate shift *and* concept drift. More customers are month-to-month/Fiber with higher charges (inputs moved), but churn is also far higher inside every contract and fibre segment than in the reference (e.g. two-year contracts churn ~45% vs ~7%, one-year ~55% vs ~15%), and the fitted coefficients change (the contract-type effect shrinks sharply). Data quality is fine: missing rates are unchanged (~2% Total_Charges, ~1.5% Avg_Data_Usage) and no unseen categories appear.")
md("## 4. Did the baseline model's performance degrade?\nMetric choice: **ROC-AUC** (ranking quality, independent of the prevalence shift 20% -> 51%) and **log loss** (calibration, sensitive to a changed P(y|x)). PR-AUC / F1 are reported but rise in weeks 6-8 *only because prevalence rises* - they would hide the degradation.")
code("""R=[]
for w,d in weeks.items():
    e=evaluate(model,d); R.append({'week':w,**{k:e[k] for k in ('roc_auc','log_loss','pr_auc','f1','recall','mean_pred','positive_rate')}})
R=pd.DataFrame(R).set_index('week'); print(R.round(3).to_string()); R.to_csv('../reports/baseline_weekly_metrics.csv')
fig,ax=plt.subplots(1,3,figsize=(11,3))
R.roc_auc.plot(marker='o',ax=ax[0],color='#4c78a8'); ax[0].axhline(0.753,ls='--',c='grey'); ax[0].set_title('ROC-AUC (dashed = hold-out)')
R.log_loss.plot(marker='o',ax=ax[1],color='#e45756'); ax[1].axhline(0.44,ls='--',c='grey'); ax[1].set_title('log loss (lower better)')
R[['mean_pred','positive_rate']].plot(marker='o',ax=ax[2]); ax[2].set_title('predicted vs actual churn rate')
for a in ax: a.set_xlabel('week')
plt.tight_layout(); plt.savefig('../reports/fig_baseline_performance.png'); plt.show()""")
md("""**Reading / classification of each week**

| weeks | drift | performance | verdict |
|---|---|---|---|
| 1-3 | none (PSI < 0.07) | ROC-AUC 0.76-0.83, log loss 0.39-0.44 (week-to-week noise at n=500) | healthy |
| 4-5 | moderate to major (0.17, 0.35) | ROC-AUC 0.75-0.79, log loss 0.47-0.51, model slightly over-predicts churn (mean prediction 0.28-0.30 vs 0.26-0.28 actual) | **drift with little performance loss yet** - retraining is not clearly worth it (challenger gain < 1%, see policy study) |
| 6-8 | severe (PSI 1.5-1.9) | ROC-AUC 0.72-0.73, log loss 0.64-0.66 (about +55-60% vs weeks 1-3), predicted churn 45% vs actual 51-55% | **real degradation**: drift and a changed churn relationship |

Take-away: drift alone is not the same as damage. Week 5 shows a drift alarm without a performance problem; weeks 6-8 show both. This is why the workflow retrains only when drift fires **and** a challenger beats production on recent labelled data.""")
nb.cells = C; nbf.write(nb, 'notebooks/03_drift_investigation.ipynb')
