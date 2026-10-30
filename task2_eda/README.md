# Task 2 — Exploratory Data Analysis (EDA)

## Objective
Explore the Titanic dataset to understand distributions, spot anomalies and find relationships
between features before any modelling.

## Dataset
Same Titanic file as Task 1 (`titanic.csv` is used if present, otherwise downloaded and cached).

## What the script does
- Summary statistics and missing-value counts (Age and Cabin have many gaps).
- Histograms of all numeric features, Age distribution + boxplot.
- Fare by passenger class; survival counts.
- **Survival rate by class and sex** (printed and plotted).
- Pairplot, correlation heatmap, skewness of Age and Fare.
- The "key observations" printed at the end are computed from the data, so they can't go stale.

## How to run
```bash
pip install -r ../requirements.txt
python task2_eda.py
```
All figures are saved to `outputs/`. `task2_eda.ipynb` is the original notebook version.

## Interview Questions & Answers

1. Purpose of EDA:
To understand data distribution, detect anomalies, and generate insights before modeling.

2. Boxplots:
Show median, quartiles, and outliers; help in comparing distributions across categories.

3. Correlation:
Measures relationship between variables; useful for feature selection and avoiding multicollinearity.

4. Detect Skewness:
Use .skew() in pandas or visualize histograms; skewed data may need transformation.

5. Multicollinearity:
When two or more independent variables are highly correlated; can affect regression models.

6. Tools for EDA:
Pandas, Matplotlib, Seaborn, Plotly, Excel, Tableau.

7. EDA in Problem Detection:
Example: Found missing Age values in Titanic dataset and decided to fill them with median.

8. Role of Visualization in ML:
Helps in understanding feature distributions, patterns, correlations, and model interpretability.
