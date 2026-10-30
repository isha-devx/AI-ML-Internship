"""AI & ML Internship - Task 2: Exploratory Data Analysis (Titanic dataset).

Data: looks for `titanic.csv` next to this script first, then downloads it.
All figures are saved to ./outputs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
        df.to_csv(local, index=False)
        return df
    except Exception as exc:
        raise SystemExit(
            f"Could not download the Titanic dataset ({exc}).\n"
            f"Download it from:\n  {URL}\nand save it as:\n  {local}"
        )


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.show()


df = load_titanic()

print("First 5 rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nDescriptive statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Histograms of all numeric columns
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols].hist(bins=20, figsize=(12, 10), edgecolor="black")
plt.suptitle("Histograms of Numerical Features", fontsize=16)
save("01_histograms.png")

# Age distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Age Distribution")
save("02_age_distribution.png")

# Age boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Age")
plt.title("Age Distribution and Outliers")
save("03_age_boxplot.png")

# Fare by class
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.title("Fare Distribution by Passenger Class")
save("04_fare_by_class.png")

# Survival count
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="Survived")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Did Not Survive", "Survived"])
plt.title("Passenger Survival Count")
save("05_survival_count.png")

# Survival rate by sex and class (a real insight, not just counts)
print("\nSurvival rate by Sex:")
print(df.groupby("Sex")["Survived"].mean().round(3))
print("\nSurvival rate by Pclass:")
print(df.groupby("Pclass")["Survived"].mean().round(3))

plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex")
plt.title("Survival Rate by Class and Sex")
plt.ylabel("Survival rate")
save("06_survival_by_class_sex.png")

# Pairplot
pair = df[["Age", "Fare", "Pclass", "Survived"]].dropna()
sns.pairplot(pair, hue="Survived", diag_kind="hist")
plt.savefig(OUT / "07_pairplot.png", dpi=120)
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True,
            cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
save("08_correlation_matrix.png")

# Skewness
print("\nSkewness:")
print(df[["Age", "Fare"]].skew().round(3))

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Fare", bins=30, kde=True)
plt.title("Fare Distribution and Skewness")
save("09_fare_skewness.png")

# Observations computed from the data (so they can never be out of date)
print("\nKey observations:")
print(f"- {len(df)} passengers; overall survival rate {df['Survived'].mean():.1%}.")
print(f"- Largest passenger class: Pclass {df['Pclass'].value_counts().idxmax()}.")
print(f"- Missing Age: {df['Age'].isna().mean():.1%}; missing Cabin: {df['Cabin'].isna().mean():.1%}.")
print(f"- Fare skewness = {df['Fare'].skew():.2f} (right-skewed, high-fare outliers).")
print("\nTask 2 EDA completed successfully!")
