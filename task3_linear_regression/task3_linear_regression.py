"""AI & ML Internship - Task 3: Linear Regression (simple & multiple).

Dataset: scikit-learn's Diabetes dataset (442 real patients, 10 baseline
features). Target = a quantitative measure of disease progression one year
after baseline. Bundled with scikit-learn, so no download is needed.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# ---- 1. Load data (scaled=False keeps original units, so coefficients are interpretable)
bunch = load_diabetes(scaled=False, as_frame=True)
X, y = bunch.data, bunch.target.rename("progression")
print("Dataset shape:", X.shape)
print(X.head())
print("\nTarget summary:")
print(y.describe().round(2))

# ---- 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")


def evaluate(name, model, cols):
    model.fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {np.sqrt(mse):.2f}")
    print(f"  R2   : {r2:.4f}")
    return pred, r2


# ---- 3. Simple linear regression (one feature: BMI)
simple = LinearRegression()
pred_simple, r2_simple = evaluate("Simple regression (bmi only)", simple, ["bmi"])
print(f"  Slope: {simple.coef_[0]:.2f} | Intercept: {simple.intercept_:.2f}")

plt.figure(figsize=(7, 5))
plt.scatter(X_test["bmi"], y_test, alpha=0.7, label="Actual")
order = np.argsort(X_test["bmi"].to_numpy())
plt.plot(X_test["bmi"].to_numpy()[order], pred_simple[order], color="red", label="Fitted line")
plt.xlabel("BMI")
plt.ylabel("Disease progression")
plt.title("Simple Linear Regression: BMI vs Progression")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "01_simple_regression.png", dpi=150)
plt.show()

# ---- 4. Multiple linear regression (all features)
multi = LinearRegression()
pred_multi, r2_multi = evaluate("Multiple regression (all 10 features)", multi, list(X.columns))

coef = pd.DataFrame({"Feature": X.columns, "Coefficient": multi.coef_}).sort_values(
    "Coefficient", key=abs, ascending=False
)
print("\nCoefficients (sorted by magnitude):")
print(coef.round(3).to_string(index=False))
print(f"Intercept: {multi.intercept_:.2f}")

# ---- 5. Actual vs predicted & residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, pred_multi, alpha=0.7)
lims = [min(y_test.min(), pred_multi.min()), max(y_test.max(), pred_multi.max())]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs Predicted")
axes[0].legend()

residuals = y_test - pred_multi
axes[1].scatter(pred_multi, residuals, alpha=0.7)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "02_multiple_regression_diagnostics.png", dpi=150)
plt.show()

# ---- 6. Multicollinearity check (VIF = 1 / (1 - R2 of feature ~ other features))
vif = {}
for col in X.columns:
    others = X.drop(columns=col)
    r2_col = LinearRegression().fit(others, X[col]).score(others, X[col])
    vif[col] = 1.0 / (1.0 - r2_col)
vif = pd.Series(vif).sort_values(ascending=False)
print("\nVariance Inflation Factors (>10 suggests strong multicollinearity):")
print(vif.round(1).to_string())

print("\nSummary")
print(f"  Simple (bmi)      R2 = {r2_simple:.3f}")
print(f"  Multiple (all)    R2 = {r2_multi:.3f}")
print("\nTask 3 Linear Regression completed successfully!")
