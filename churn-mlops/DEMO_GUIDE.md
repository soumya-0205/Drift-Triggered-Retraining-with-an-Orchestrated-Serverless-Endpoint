# Live demo guide (run it on your laptop, ~5 minutes)

## A. One-time setup (do this the day before)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-demo.txt
python demo/demo.py --auto                                 # full rehearsal, ~50 s, no pauses. It must end with "DONE".
```
Tested on Linux with Python 3.12. **Not tested on Windows/macOS** - if the endpoint step fails on Windows, run the demo inside WSL, or fall back to section D.

## B. Optional: show the real orchestrator and registry UIs (recommended - the brief asks for the graph)
Open two extra terminals in the project folder (same virtual env):
```bash
prefect server start                                       # terminal 1  -> http://127.0.0.1:4200
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # terminal 2 -> http://127.0.0.1:5000  (pre-built registry: v1 baseline, v2 rejected, v3 production)
```
In the **demo terminal**, set the API URL *before* starting, so the live runs appear in the Prefect UI:
`export PREFECT_API_URL=http://127.0.0.1:4200/api` (Windows PowerShell: `$env:PREFECT_API_URL="http://127.0.0.1:4200/api"`).

## C. Presenting (interactive mode: `python demo/demo.py`, press Enter between steps)
| time | step on screen | what to say / point at (rubric item) |
|---|---|---|
| 0:00 | intro | Problem in one sentence: detect drift, retrain only if worth it, never promote a worse model, escalate to a human when automation shouldn't be trusted. |
| 0:30 | Step 1 baseline | Logistic regression, hold-out ROC-AUC 0.753; metric choice for 20% churn: PR-AUC/F1 reported, ROC-AUC + log loss monitored. Open MLflow UI -> churn-model: model + metrics + `reference_profile.json`. (*Baseline & registry*) |
| 1:15 | Step 2 endpoint | The function reads the registry alias `production`; no model path in code. Cold ~2.7 s vs warm ~6 ms (`reports/latency.json`). Bad input -> HTTP 400. (*Serverless endpoint*) |
| 2:00 | Step 3 eight weeks | Narrate as it prints: weeks 1-4 no action; **week 5** PSI 0.35 -> challenger trained but only +0.8% < 2% gate -> *rejected*; **week 6** PSI 1.54 -> **ESCALATION, pipeline paused**. Now *you* type `approve` as the human on call. **Week 7** retrain, +8.7% -> PROMOTED and the endpoint line changes v1 -> v3 with no restart. Show Prefect UI: week 6 graph (escalation artifact, skipped branches) and week 7 (retrain -> compare -> register -> promote). (*Workflow, escalation, model switching*) |
| 3:45 | Step 4 faults | Corrupt batch fails at validation with no retry; forced-worse challenger is rejected by the gate; production and endpoint answer unchanged both times. Retries (1-2-4 s backoff) are for registry/IO faults - see `logs/fault_injection.log`. (*Failure handling*) |
| 4:20 | Step 5 policy | Table: never retrain is worst; drift PSI>=0.25 is cheapest with the same quality as weekly; add escalation at PSI>=1.0 for oversight (+3% cost). Timeline chart opens automatically. (*Policy study*) |
| 4:50 | close | Escalation rule: stop if max PSI >= 1.0 (2.9x the largest benign value 0.345; below the smallest severe week 1.54) or if 2 consecutive challengers fail the gate while drift persists. |

## D. If something breaks on the day
1. `python demo/demo.py --auto` prints the same story without pauses.
2. Show the saved evidence instead: `evidence/prefect_graph_week6.png`, `prefect_graph_week7.png`, `run_history_table.png`, `mlflow_registry_versions.png`, `logs/endpoint_promotion_demo.log`, `logs/fault_injection.log`.
3. Port in use? The demo picks a free port itself. Prefect "database is locked" noise at start is harmless - rerun.

## E. Likely questions (answers from our results)
* *Why logistic regression?* Boosted trees overfit 2,250 rows (0.91 train vs 0.72 hold-out ROC-AUC); LR got 0.753 and is stable when retrained on small recent windows.
* *Was week 5 drift harmful?* Little: ROC-AUC 0.75 (at hold-out level), log loss +23%; the gated challenger gained only 0.8%, so not retraining was right. Weeks 6-8 are real degradation (log loss +54-61%, churn 21% -> 51-55%, relationship changed).
* *How is look-ahead prevented?* The week-N run only reads weeks 1..N (`LookAheadError` otherwise, tested); challengers are compared on eval halves of the last two weeks that nobody trained on.
* *How does the endpoint pick the model?* Registry alias `production`, re-checked every 2 s, reloaded only if the version changed; promotion = moving the alias, only inside the gated workflow.
* *Why escalate at 1.0, not 0.25 or 2.0?* 0.25 also pages on week 5 where the gate already rejected the challenger; 2.0 never fires (max observed 1.90). Anything between 0.35 and 1.5 behaves the same on this data.
* *Is the human real?* In the 8-week evidence run the human step is simulated; in the live demo **you** are the human.
* *What costs did you assume?* Missed churner $200, false alert $20, retraining run $50 (sensitivity up to $5,000 in the report).
