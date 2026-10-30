"""Disease outbreak prediction - training & evaluation pipeline.

Predicts whether a region will have an outbreak NEXT week (incidence >= 60 per
100k) from this week's weather, demographic and surveillance features.

Key design choices
  * Time-based split: train on the first 4 years, test on the final year, so
    the model is never evaluated on the "past" of the data it learned from.
  * One sklearn Pipeline (median imputation + scaling + one-hot encoding +
    model), so preprocessing is fitted on training data only (no leakage).
  * Models compared with TimeSeriesSplit cross-validation (ROC-AUC); the
    winner is refit and evaluated once on the held-out test year.
  * Reported against a naive baseline ("outbreak now => outbreak next week").

Usage
    python main.py                     # uses data/outbreak_data.csv (auto-generated if missing)
    python main.py --data my_data.csv  # your own CSV with the same columns
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_data import DATA_PATH, OUTBREAK_THRESHOLD, simulate

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
MODEL_DIR = HERE / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "outbreak_next_week"
CATEGORICAL = ["region"]
NUMERIC = [
    "week_of_year", "temperature_c", "humidity_pct", "rainfall_mm",
    "population_density", "vaccination_rate_pct", "mobility_index",
    "infection_rate", "infection_rate_prev", "infection_rate_change",
]
TEST_YEARS = 1  # final year held out for testing


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        if path != DATA_PATH:
            raise SystemExit(f"Data file not found: {path}")
        print(f"{path} not found - generating simulated data...")
        path.parent.mkdir(exist_ok=True)
        simulate().to_csv(path, index=False)
    df = pd.read_csv(path)
    required = {"region", "week_index", TARGET, "infection_rate", "infection_rate_prev"}
    missing = (required | (set(NUMERIC) - {"infection_rate_change"})) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")
    df["infection_rate_change"] = df["infection_rate"] - df["infection_rate_prev"]
    return df.sort_values(["week_index", "region"]).reset_index(drop=True)


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", preprocess), ("model", model)])


def get_feature_names(pipe: Pipeline) -> list:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def main(data_path: Path, seed: int = 42) -> None:
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows | {df['region'].nunique()} regions | "
          f"{df['week_index'].nunique()} weeks")
    print(f"Outbreak rate (target = 1): {df[TARGET].mean():.1%}")
    print("Missing values per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string())

    # ---- Exploration plot (temperature vs infection rate) ----------------
    plt.figure(figsize=(7, 5))
    plt.scatter(df["temperature_c"], df["infection_rate"], c=df[TARGET],
                cmap="coolwarm", alpha=0.35, s=14)
    plt.axhline(OUTBREAK_THRESHOLD, color="black", linestyle="--", linewidth=1,
                label=f"Outbreak threshold ({OUTBREAK_THRESHOLD:.0f}/100k)")
    plt.yscale("log")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Infection rate (cases per 100k, log scale)")
    plt.title("Temperature vs Infection Rate (colour = outbreak next week)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "01_temperature_vs_infection.png", dpi=150)
    plt.close()

    # ---- Time-based split ------------------------------------------------
    cutoff = df["week_index"].max() + 1 - 52 * TEST_YEARS
    train, test = df[df["week_index"] < cutoff], df[df["week_index"] >= cutoff]
    X_train, y_train = train[NUMERIC + CATEGORICAL], train[TARGET]
    X_test, y_test = test[NUMERIC + CATEGORICAL], test[TARGET]
    print(f"\nTrain: {len(train)} rows (weeks < {cutoff}) | Test: {len(test)} rows (weeks >= {cutoff})")

    # ---- Model comparison via time-series cross-validation ------------------
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    cv_auc = {}
    print("\nTime-series cross-validation (ROC-AUC, training data only):")
    for name, model in candidates.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train, cv=tscv, scoring="roc_auc")
        cv_auc[name] = float(scores.mean())
        print(f"  {name:<20} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_auc, key=cv_auc.get)
    print(f"\nBest model by CV: {best_name}")

    # ---- Final evaluation on held-out year -----------------------------------
    fitted, results = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fitted[name] = pipe
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

    # Baselines
    majority = int(y_train.mode()[0])
    baseline_maj = accuracy_score(y_test, np.full(len(y_test), majority))
    persist_pred = (test["infection_rate"] >= OUTBREAK_THRESHOLD).astype(int)
    results["Baseline: persistence"] = {
        "accuracy": accuracy_score(y_test, persist_pred),
        "precision": precision_score(y_test, persist_pred, zero_division=0),
        "recall": recall_score(y_test, persist_pred, zero_division=0),
        "f1": f1_score(y_test, persist_pred, zero_division=0),
        "roc_auc": float("nan"),
    }

    table = pd.DataFrame(results).T.round(4)
    print("\nHeld-out test results (final year):")
    print(table.to_string())
    print(f"Majority-class baseline accuracy: {baseline_maj:.4f}")

    best = fitted[best_name]
    best_proba = best.predict_proba(X_test)[:, 1]
    best_pred = (best_proba >= 0.5).astype(int)
    print(f"\nClassification report - {best_name}:")
    print(classification_report(y_test, best_pred, target_names=["No outbreak", "Outbreak"]))

    # ---- Plots ----------------------------------------------------------------------
    cm = confusion_matrix(y_test, best_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No outbreak", "Outbreak"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "02_confusion_matrix.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC Curves - held-out test year")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "03_roc_curves.png", dpi=150)
    plt.close()

    perm = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=seed, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=NUMERIC + CATEGORICAL).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Permutation Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "04_feature_importance.png", dpi=150)
    plt.close()
    print("Top features:")
    print(importance.sort_values(ascending=False).head(5).round(4).to_string())

    # ---- Save model + metrics --------------------------------------------------------
    joblib.dump(best, MODEL_DIR / "outbreak_model.joblib")
    metrics = {
        "best_model": best_name,
        "cv_roc_auc": cv_auc,
        "test": {k: {m: (None if np.isnan(v) else round(float(v), 4)) for m, v in d.items()}
                 for k, d in results.items()},
        "majority_baseline_accuracy": round(float(baseline_maj), 4),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "data_is_simulated": data_path == DATA_PATH,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {MODEL_DIR / 'outbreak_model.joblib'}")
    print(f"Metrics saved to {OUT / 'metrics.json'}")
    print("Disease outbreak prediction completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the outbreak prediction model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the CSV dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data, args.seed)
