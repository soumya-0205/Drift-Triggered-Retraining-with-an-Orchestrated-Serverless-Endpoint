import os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
T = Path(tempfile.mkdtemp(prefix="pytest_"))
os.environ.update(PIPELINE_STATE_DIR=str(T / "state"), PIPELINE_REPORT_DIR=str(T / "reports"), HUMAN_DECISIONS=str(T / "human.json"),
                  MLFLOW_TRACKING_URI=f"sqlite:///{T}/mlflow.db", ARTIFACT_ROOT=str(T / "art"), PREFECT_HOME=str(T / "prefect"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "flows")]
