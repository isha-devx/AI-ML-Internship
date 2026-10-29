# Task 5: Decision Trees & Random Forests
# AI/ML Internship

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


def main():
    # Load dataset
    # The CSV file includes a markdown-style description block before and after
    # the actual tabular data, so we skip the metadata rows and keep only rows
    # that actually contain valid numeric target values.
    dataset = pd.read_csv("dataset.csv", skiprows=[0, 1], on_bad_lines="skip")
    dataset = dataset.dropna(subset=["target"])
    dataset = dataset.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    X = dataset.drop(columns=["target"])
    y = dataset["target"]

    print("Dataset loaded successfully!")
    print("Dataset shape:", dataset.shape)
    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nTarget distribution:")
    print(y.value_counts())

    # Split data. This dataset is extremely small and imbalanced, so fall back to a
    # deterministic split when sklearn's stratified split would fail.
    if y.nunique() < 2 or len(y) < 10:
        print("\nDataset is too small for a standard stratified split; using a deterministic fallback split.")
        minority_idx = y[y == y.min()].index[0]
        majority_idx = y[y == y.max()].index[0]
        test_idx = [minority_idx, majority_idx]
        X_train = X.drop(index=test_idx)
        X_test = X.loc[test_idx]
        y_train = y.drop(index=test_idx)
        y_test = y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt_model.fit(X_train, y_train)

    y_pred_dt = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)

    try:
        dt_cv = cross_val_score(dt_model, X, y, cv=min(3, len(y))).mean()
    except ValueError:
        dt_cv = None

    print("\nDecision Tree Model")
    print("---------------------")
    print(f"Accuracy: {dt_accuracy * 100:.2f}%")
    if dt_cv is not None:
        print(f"Cross-validation accuracy: {dt_cv * 100:.2f}%")
    else:
        print("Cross-validation accuracy: not available for this tiny dataset")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dt, target_names=["No Heart Disease", "Heart Disease"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))

    plt.figure(figsize=(18, 10))
    plot_tree(
        dt_model,
        feature_names=X.columns,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()

    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    try:
        rf_cv = cross_val_score(rf_model, X, y, cv=min(3, len(y))).mean()
    except ValueError:
        rf_cv = None

    print("\nRandom Forest Model")
    print("-------------------")
    print(f"Accuracy: {rf_accuracy * 100:.2f}%")
    if rf_cv is not None:
        print(f"Cross-validation accuracy: {rf_cv * 100:.2f}%")
    else:
        print("Cross-validation accuracy: not available for this tiny dataset")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=["No Heart Disease", "Heart Disease"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Feature importance
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    feature_importance.head(10).plot(kind="bar", color="steelblue")
    plt.title("Top 10 Features by Importance (Random Forest)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    # Compare both models
    print("\nModel Comparison")
    print("----------------")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print(f"Random Forest is better: {rf_accuracy > dt_accuracy}")

    print("\nTask 5 completed successfully!")


if __name__ == "__main__":
    main()
