# Drift-triggered retraining for a churn model (MLOps Project 3)

Open-source, free-tier stack: **MLflow** (tracking + model registry, SQLite backend) · **Prefect 3** (orchestration, stand-in for AWS Step Functions)
· **functions-framework** (serverless-style HTTP function, stand-in for Cloud Functions/Lambda) · scikit-learn · scipy.

## Run it
```bash
pip install prefect mlflow functions-framework scikit-learn pandas scipy matplotlib nbformat nbconvert ipykernel
cd src && python train_baseline.py                 # Task 1: baseline -> registry v1 -> alias "production"
cd ../notebooks && python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace 03_drift_investigation.ipynb   # Task 3
cd ../src && python latency_study.py               # Task 2: 30 cold + 200 warm calls -> reports/latency.json
cd ../flows && python run_eight_weeks.py           # Tasks 4+7: 8 weekly flow runs -> reports/run_summaries.jsonl, escalations.jsonl
cd ../src && python simulate.py && python policy_report.py       # Task 5: 5 retraining policies
cd ../flows && python fault_tests.py               # Task 6 (isolated registry) -> reports/fault_injection.json
python endpoint_promotion_demo.py                  # live HTTP endpoint switches version after promotion, no restart
# serve locally:  functions-framework --source function/main.py --target predict --port 8080
# Prefect UI:     PREFECT_HOME=$PWD/.prefect prefect server start   (then run flows with PREFECT_API_URL=http://127.0.0.1:4200/api)
```

## Live demo
`python demo/demo.py` (guided, ~5 min) or `--auto`; setup, UI instructions, talk track and Q&A in `DEMO_GUIDE.md` (minimal packages: `requirements-demo.txt`).

## Extra tooling
* **Evidently** cross-check of my PSI/KS code: `cd src && python evidently_crosscheck.py` -> `reports/evidently_*.csv/json`, HTML report, screenshot in `evidence/`.
* **pytest:** `python -m pytest tests -q` (22 tests: look-ahead guard, validation, gate, escalation, reproducibility, drift statistics).
* **Dockerfile** for the endpoint and **GitHub Actions** CI (`.github/workflows/ci.yml`) - written but not executed in the authoring sandbox.

## Evidence map (deliverables)
| deliverable | where |
|---|---|
| registry evidence (baseline + promoted) | `evidence/mlflow_registry_*.png`, `reports/registry_export.json` |
| orchestrator graph + 8-week run history | `evidence/prefect_graph_week{5,6,7}.png`, `evidence/prefect_run_history.png`, `evidence/run_history_table.png`, `reports/run_history_table.csv`, `reports/prefect_run_history.json`, `reports/run_summaries.jsonl` |
| drift notebook | `notebooks/03_drift_investigation.ipynb` |
| policy table + timeline chart | `reports/policy_*.csv`, `reports/fig_policy_timeline.png` |
| fault-injection logs | `logs/fault_injection.log`, `reports/fault_injection.json` |
| escalation definition + firing | `reports/REPORT.pdf` section 7, `reports/escalations.jsonl`, `evidence/prefect_graph_week6.png` |
| reproducibility check | `reports/reproducibility_check.json` |

## Layout
| path | what |
|---|---|
| `configs/pipeline_config.json` | all thresholds, gate, window and cost settings (read by the code) |
| `src/common.py` | features, model pipeline (scaled logistic regression), metrics, PSI/KS drift code, reference profile |
| `src/pipeline_core.py` | pure decision logic: validation, weekly train/eval split, challenger training, promotion gate, cost model |
| `src/registry.py` | MLflow wrapper; `promote()` only sets alias `production` |
| `flows/weekly_pipeline.py` | the Prefect flow (one run = one week) |
| `function/main.py` | the prediction endpoint (resolves alias `production`) |
| `src/simulate.py` | offline policy simulator using the *same* decision code as the flow |
| `reports/`, `logs/` | all results, run summaries, escalation records, fault-injection log |

## Design decisions (all settings live in `configs/pipeline_config.json`)
* **Model:** standardised + one-hot logistic regression. Gradient boosting over-fitted the 3k-row reference (train ROC-AUC 0.90 vs 0.72 on hold-out) and generalised worse to weeks 1-3; LR was best on the hold-out (0.753) and is stable under retraining on small recent windows.
* **Metrics:** class balance is ~20% churn, so PR-AUC/F1 at a fixed threshold (0.35) are reported, but *monitoring* uses **ROC-AUC** and **log loss**: weeks 6-8 have ~51-55% churn, which inflates PR-AUC/F1 and would hide degradation.
* **Reference profile** (bin edges, moments, category frequencies, missing rates, 1.5k-row sample for KS) is stored in MLflow *with* each model version, so drift is always measured against the data the serving model was trained on. After a promotion the profile is re-baselined automatically.
* **No look-ahead:** the flow for week N only calls `load_week(k<=N)`; `ingest` raises `LookAheadError` otherwise (tested). Each weekly batch is split 60/40 (train half / eval half, fixed seed). Challengers train on reference-train (weight 0.25) + train halves of the last 3 weeks and are compared with production on the **eval halves of the last 2 weeks** - rows no model ever trained on.
* **Promotion gate:** challenger must cut log loss by >=2% and not lose >0.01 ROC-AUC vs production on that recent eval data. Otherwise it is registered (audit trail) but never promoted.
* **Drift trigger:** max feature PSI >= 0.25. **Escalation:** (a) max PSI >= 1.0, or (b) 2 consecutive challengers rejected while drift persists -> pipeline pauses (no automatic retrain/promotion), writes `reports/escalations.jsonl`, a CRITICAL log line and a Prefect artifact; it resumes only when `configs/human_decisions.json` holds a decision for that week.
* **Errors:** deterministic failures (bad schema) are *not* retried; infrastructure errors (registry/IO) retry with 1-2-4 s backoff; any failure leaves production untouched and is recorded as `FAILED` with the error.

## Honest limitations
* Everything runs locally; "serverless" = functions-framework handler (portable to Cloud Functions/Lambda), not a cloud deployment. Cold-start numbers are local process start-ups, not cloud numbers.
* The human step in the 8-week run is **simulated** by the driver (clearly labelled in `configs/human_decisions.json`).
* Weekly batches are 500 rows; gate and policy differences of a few percent are within sampling noise. The policy study is one deterministic run, not a distribution.
* MLflow version numbers include rejected challengers (v2 = week-5 challenger that failed the gate; v3 = week-7 promoted model).
* No Prefect UI screenshot is included (no browser here); start the server as above to take one. Prefect prints a harmless telemetry error when offline.
