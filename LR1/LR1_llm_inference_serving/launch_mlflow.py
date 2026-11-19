import pathlib
import sqlite3
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
with sqlite3.connect(root / "mlflow.db") as db:
    exp = db.execute("SELECT experiment_id FROM experiments WHERE name = ?", ("LR1_llm_inference_serving",)).fetchone()
    if exp is None:
        raise SystemExit("LR1 experiment not found")
    uri = (root / "mlartifacts").as_uri()
    db.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?", (uri, exp[0]))
    for (run_id,) in db.execute("SELECT run_uuid FROM runs WHERE experiment_id = ?", exp):
        db.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (f"{uri}/{run_id}/artifacts", run_id))
subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
                "sqlite:///mlflow.db", "--host", "127.0.0.1", "--port", "5002", "--workers", "1"],
               cwd=root, check=True)
