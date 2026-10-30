"""AI & ML Internship - Task 1: Data Cleaning & Preprocessing (Titanic dataset).

Steps: inspect -> handle missing values -> encode categoricals -> scale
numerical features -> detect/remove outliers.

Data: looks for `titanic.csv` next to this script first, then downloads it.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_titanic() -> pd.DataFrame:
    local = HERE / "titanic.csv"
    if local.exists():
        return pd.read_csv(local)
    try:
        df = pd.read_csv(URL)
        df.to_csv(local, index=False)  # cache so later runs work offline
        return df
    except Exception as exc:  # network / URL problem
        raise SystemExit(
            f"Could not download the Titanic dataset ({exc}).\n"
            f"Download it from:\n  {URL}\nand save it as:\n  {local}"
        )


df = load_titanic()

print("First 5 rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nSummary statistics:")
print(df.describe(include="all"))

# ---- Step 2: missing values ------------------------------------------------
print("\nMissing values before handling:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])  # ~77% missing

print("\nMissing values after handling:")
print(df.isnull().sum())

# ---- Step 3: encode categoricals --------------------------------------------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # binary -> 0/1
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)
print("\nEncoded sample:")
print(df.head())

# ---- Step 4: standardise numerical features ----------------------------------
num_cols = ["Age", "Fare"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
print("\nScaled numerical features (mean ~0, std ~1):")
print(df[num_cols].describe().loc[["mean", "std"]].round(3))

# ---- Step 5: outlier detection -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=df["Age"], ax=axes[0]).set_title("Boxplot of Age (scaled)")
sns.boxplot(y=df["Fare"], ax=axes[1]).set_title("Boxplot of Fare (scaled)")
plt.tight_layout()
plt.savefig(OUT / "boxplots_before_outlier_removal.png", dpi=150)
plt.show()

# ---- Step 6: remove extreme Fare outliers (IQR rule) ---------------------------
q1, q3 = df["Fare"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
before = len(df)
df = df[df["Fare"] <= upper].copy()
print(f"\nRows removed as Fare outliers: {before - len(df)} (of {before})")
print("Shape after outlier removal:", df.shape)
print("Remaining missing values:", int(df.isnull().sum().sum()))

df.to_csv(OUT / "titanic_cleaned.csv", index=False)
print(f"\nCleaned dataset saved to {OUT / 'titanic_cleaned.csv'}")
print("Task 1 completed successfully!")
