"""Simulate weekly disease-surveillance data for several regions.

WHY SIMULATED?  Real outbreak surveillance data is not freely downloadable in
one clean file, so this script builds a transparent, reproducible stand-in.
It is NOT real epidemiological data - treat every result as a demonstration of
the ML workflow, not as an epidemiological finding.  To use real data, give
`main.py` a CSV with the same columns (see README).

Simulation idea (a simple stochastic model, not a full SIR model):
  * Weather follows yearly seasonal cycles plus noise.
  * Weekly incidence (cases per 100k) is auto-regressive on the log scale and
    is pushed up by humidity, rainfall, population density and mobility, and
    pushed down by vaccination coverage.
  * Label `outbreak_next_week` = 1 if NEXT week's incidence >= OUTBREAK_THRESHOLD.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUTBREAK_THRESHOLD = 60.0  # cases per 100k that counts as an outbreak
DATA_PATH = Path(__file__).resolve().parent / "data" / "outbreak_data.csv"


def simulate(n_regions: int = 10, n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_weeks = n_years * 52
    rows = []

    for r in range(n_regions):
        base_temp = rng.uniform(16, 30)
        temp_amp = rng.uniform(3, 9)
        base_hum = rng.uniform(50, 75)
        density = float(np.exp(rng.uniform(np.log(80), np.log(6000))))
        base_vax = rng.uniform(45, 85)
        base_mobility = rng.uniform(80, 100)
        phase = rng.uniform(0, 52)
        base_log_inc = rng.uniform(3.7, 4.0)

        log_inc = base_log_inc
        incidence = [np.exp(log_inc)]
        weather = []
        for t in range(n_weeks):
            season = np.sin(2 * np.pi * (t + phase) / 52)
            temp = base_temp + temp_amp * season + rng.normal(0, 1.5)
            hum = np.clip(base_hum + 12 * season + rng.normal(0, 6), 20, 100)
            rain = max(0.0, 60 + 45 * season + rng.normal(0, 25))
            vax = np.clip(base_vax + 0.04 * t + rng.normal(0, 1.0), 0, 100)
            mob = base_mobility + 6 * np.sin(2 * np.pi * (t + phase / 2) / 52) + rng.normal(0, 4)
            weather.append((temp, hum, rain, vax, mob))

            pressure = (
                0.040 * (hum - 62)
                + 0.008 * (rain - 60)
                + 0.030 * (mob - 90)
                + 0.05 * (np.log(density) - 6.5)
                - 0.020 * (vax - 65)
                + 0.035 * (temp - base_temp)
            )
            log_inc = 0.45 * log_inc + 0.55 * base_log_inc + pressure + rng.normal(0, 0.55)
            incidence.append(np.exp(log_inc))

        for t in range(n_weeks):
            temp, hum, rain, vax, mob = weather[t]
            rows.append(
                {
                    "region": f"Region_{r + 1:02d}",
                    "week_index": t,
                    "week_of_year": t % 52 + 1,
                    "temperature_c": round(temp, 2),
                    "humidity_pct": round(hum, 2),
                    "rainfall_mm": round(rain, 2),
                    "population_density": round(density, 1),
                    "vaccination_rate_pct": round(vax, 2),
                    "mobility_index": round(mob, 2),
                    "infection_rate": round(incidence[t], 2),  # cases / 100k, this week
                    "infection_rate_prev": round(incidence[t - 1], 2) if t > 0 else np.nan,
                    "outbreak_next_week": int(incidence[t + 1] >= OUTBREAK_THRESHOLD),
                }
            )

    df = pd.DataFrame(rows).sort_values(["week_index", "region"]).reset_index(drop=True)

    # Realistic data problems: ~3% of weather sensor readings are missing.
    for col in ["temperature_c", "humidity_pct", "rainfall_mm"]:
        df.loc[rng.random(len(df)) < 0.03, col] = np.nan
    return df


if __name__ == "__main__":
    data = simulate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(data)} rows to {DATA_PATH}")
    print("Outbreak rate:", round(data["outbreak_next_week"].mean(), 3))
