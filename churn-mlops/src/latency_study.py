"""Task 2: cold-start vs warm latency. Cold = a brand-new process each time (imports + registry lookup + model load + first request).
Warm = repeated requests to an already-initialised instance."""
import subprocess, sys, json, time, statistics as st, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROBE = {"instances": [{"Tenure_Months": 5, "Contract_Type": "Month-to-month", "Internet_Service": "Fiber", "Monthly_Charges": 85.0,
                        "Total_Charges": 400.0, "Support_Calls_Last_Month": 3, "Avg_Data_Usage_GB": 120.0, "Late_Payments": 1}]}
CODE = f'''
import time; t0=time.perf_counter()
import sys, json; sys.path.insert(0, "{ROOT/'function'}")
import main; t_import=time.perf_counter()
from flask import Flask
app = main.predict.__globals__  # noqa
import functions_framework
c = functions_framework.create_app(target="predict", source="{ROOT/'function'/'main.py'}").test_client()
t_app=time.perf_counter()
r = c.post("/", json={json.dumps(PROBE)}); t_first=time.perf_counter()
r2 = c.post("/", json={json.dumps(PROBE)}); t_second=time.perf_counter()
print(json.dumps(dict(import_s=t_import-t0, app_s=t_app-t_import, first_request_s=t_first-t_app, total_cold_s=t_first-t0, second_request_s=t_second-t_first, status=r.status_code, version=r.get_json()["model_version"])))
'''
def pct(x, q): x = sorted(x); return x[min(len(x) - 1, int(round(q * (len(x) - 1))))]
def main(n_cold=30, n_warm=200):
    cold = []
    for _ in range(n_cold):
        t = time.perf_counter(); o = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
        wall = time.perf_counter() - t
        d = json.loads(o.stdout.strip().splitlines()[-1]); d["process_wall_s"] = wall; cold.append(d)
    sys.path.insert(0, str(ROOT / "function")); import main as fm, functions_framework
    c = functions_framework.create_app(target="predict", source=str(ROOT / "function" / "main.py")).test_client()
    c.post("/", json=PROBE); warm = []
    for _ in range(n_warm):
        t = time.perf_counter(); r = c.post("/", json=PROBE); warm.append(time.perf_counter() - t); assert r.status_code == 200
    ms = lambda v: round(v * 1000, 1)
    res = {"n_cold": n_cold, "n_warm": n_warm,
           "cold_total_ms": {"p50": ms(pct([d["total_cold_s"] for d in cold], .5)), "p95": ms(pct([d["total_cold_s"] for d in cold], .95)), "mean": ms(st.mean(d["total_cold_s"] for d in cold))},
           "cold_breakdown_mean_ms": {k: ms(st.mean(d[k] for d in cold)) for k in ("import_s", "app_s", "first_request_s")},
           "cold_process_wall_ms_p50": ms(pct([d["process_wall_s"] for d in cold], .5)),
           "warm_ms": {"p50": ms(pct(warm, .5)), "p95": ms(pct(warm, .95)), "mean": ms(st.mean(warm))}}
    res["cold_to_warm_ratio_p50"] = round(res["cold_total_ms"]["p50"] / res["warm_ms"]["p50"], 1)
    (ROOT / "reports" / "latency.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
if __name__ == "__main__": main()
