"""Predict next-week outbreak risk for one region using the saved model.

Example (high-risk looking conditions):
    python predict.py --example high
Custom input:
    python predict.py --region Region_03 --week-of-year 30 --temperature 31 \
        --humidity 85 --rainfall 140 --density 3500 --vaccination 50 \
        --mobility 105 --infection-rate 55 --infection-rate-prev 40
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "outbreak_model.joblib"

EXAMPLES = {
    "high": dict(region="Region_03", week_of_year=30, temperature_c=31, humidity_pct=85,
                 rainfall_mm=140, population_density=3500, vaccination_rate_pct=50,
                 mobility_index=105, infection_rate=55, infection_rate_prev=40),
    "low": dict(region="Region_02", week_of_year=5, temperature_c=18, humidity_pct=45,
                rainfall_mm=20, population_density=300, vaccination_rate_pct=85,
                mobility_index=85, infection_rate=12, infection_rate_prev=14),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example", choices=EXAMPLES, help="Use a built-in example input")
    p.add_argument("--region", default="Region_01")
    p.add_argument("--week-of-year", type=int, default=26)
    p.add_argument("--temperature", type=float, default=25.0)
    p.add_argument("--humidity", type=float, default=65.0)
    p.add_argument("--rainfall", type=float, default=60.0)
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--vaccination", type=float, default=65.0)
    p.add_argument("--mobility", type=float, default=90.0)
    p.add_argument("--infection-rate", type=float, default=30.0)
    p.add_argument("--infection-rate-prev", type=float, default=30.0)
    a = p.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python main.py` first to train it.")

    if a.example:
        row = EXAMPLES[a.example].copy()
    else:
        row = dict(region=a.region, week_of_year=a.week_of_year, temperature_c=a.temperature,
                   humidity_pct=a.humidity, rainfall_mm=a.rainfall, population_density=a.density,
                   vaccination_rate_pct=a.vaccination, mobility_index=a.mobility,
                   infection_rate=a.infection_rate, infection_rate_prev=a.infection_rate_prev)
    row["infection_rate_change"] = row["infection_rate"] - row["infection_rate_prev"]

    model = joblib.load(MODEL_PATH)
    prob = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    print("Input:", {k: v for k, v in row.items() if k != "infection_rate_change"})
    print(f"\nPredicted probability of an outbreak next week: {prob:.1%}")
    print("Prediction:", "OUTBREAK LIKELY" if prob >= 0.5 else "No outbreak expected")
    print("(Demo model trained on simulated data - not for real-world decisions.)")


if __name__ == "__main__":
    main()
