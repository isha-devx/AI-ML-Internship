"""Smoke test: run every project script and report PASS / FAIL.

    python run_all.py

Figures are drawn off-screen (no windows pop up). Tasks 1 and 2 need the Titanic
dataset; if it is missing and cannot be downloaded they are reported as SKIPPED.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ("task1_data_preprocessing", "task1_data_preprocessing.py"),
    ("task2_eda", "task2_eda.py"),
    ("task3_linear_regression", "task3_linear_regression.py"),
    ("task4_logistic_regression", "task4_logistic_regression.py"),
    ("task5_decision_tree_random_forest", "task5_decision_tree_rf.py"),
    ("task6_knn_classification", "knn_classification.py"),
    ("disease_outbreak_prediction", "main.py"),
]

env = {**os.environ, "MPLBACKEND": "Agg"}
failed = False
for folder, script in SCRIPTS:
    proc = subprocess.run([sys.executable, script], cwd=ROOT / folder, env=env,
                          capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if proc.returncode == 0:
        status = "PASS"
    elif "Could not download the Titanic" in text:
        status = "SKIPPED (Titanic dataset unavailable - see task README)"
    else:
        status, failed = "FAIL", True
    print(f"{status:<8} {folder}/{script}")
    if status == "FAIL":
        print("\n".join(text.strip().splitlines()[-8:]))
sys.exit(1 if failed else 0)
