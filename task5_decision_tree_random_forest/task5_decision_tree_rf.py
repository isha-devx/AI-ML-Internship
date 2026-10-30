# Task 5: Decision Trees and Random Forests
# AI/ML Internship

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    # 1. Load the dataset
    csv_path = Path(__file__).resolve().parent / "dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # The corrected CSV already contains headers on its first line.
    dataset = pd.read_csv(csv_path, encoding="utf-8-sig")
    dataset.columns = dataset.columns.str.strip()

    if "target" not in dataset.columns:
        raise ValueError(
            "The CSV must contain a 'target' column. "
            "Check that the first row contains the column headers."
        )

    # 2. Clean the data
    dataset = dataset.replace("?", pd.NA)

    for column in dataset.columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.dropna().reset_index(drop=True)

    if dataset.empty:
        raise ValueError("The dataset contains no valid rows.")

    X = dataset.drop(columns=["target"])
    y = dataset["target"].astype(int)

    if y.nunique() < 2:
        raise ValueError("The dataset must contain at least two target classes.")

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two records for "
            "a stratified train/test split."
        )

    print("Dataset loaded successfully!")
    print("Dataset shape:", dataset.shape)
    print("\nFirst five rows:")
    print(dataset.head())

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # 3. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 4. Train and evaluate the Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )
    dt_model.fit(X_train, y_train)

    y_pred_dt = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)

    print("\nDecision Tree Results")
    print("---------------------")
    print(f"Accuracy: {dt_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred_dt,
            labels=dt_model.classes_,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_dt,
        labels=dt_model.classes_,
        cmap="Blues",
    )
    plt.title("Decision Tree - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "dt_confusion_matrix.png", dpi=150)
    plt.show()

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=X.columns.tolist(),
        class_names=[str(label) for label in dt_model.classes_],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig(OUT / "decision_tree.png", dpi=150)
    plt.show()

    # 5. Train and evaluate the Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print("\nRandom Forest Results")
    print("---------------------")
    print(f"Accuracy: {rf_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred_rf,
            labels=rf_model.classes_,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_rf,
        labels=rf_model.classes_,
        cmap="Greens",
    )
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "rf_confusion_matrix.png", dpi=150)
    plt.show()

    # 6. Cross-validation on training data only
    min_class_count = int(y_train.value_counts().min())
    cv_folds = min(5, min_class_count)

    if cv_folds >= 2:
        dt_cv = cross_val_score(
            DecisionTreeClassifier(
                max_depth=3,
                random_state=42,
            ),
            X_train,
            y_train,
            cv=cv_folds,
            scoring="accuracy",
        )

        rf_cv = cross_val_score(
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
            ),
            X_train,
            y_train,
            cv=cv_folds,
            scoring="accuracy",
        )

        print("\nCross-Validation Results")
        print("------------------------")
        print(f"Decision Tree: {dt_cv.mean() * 100:.2f}%")
        print(f"Random Forest: {rf_cv.mean() * 100:.2f}%")
    else:
        print("\nCross-validation skipped: too few samples per class.")

    # 7. Feature importance
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    fig, ax = plt.subplots(figsize=(10, 6))  # explicit figure: never draw over an earlier plot
    feature_importance.head(10).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Features - Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.show()

    # 8. Compare the models
    print("\nModel Comparison")
    print("----------------")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

    if rf_accuracy > dt_accuracy:
        print("Random Forest performed better on this test set.")
    elif dt_accuracy > rf_accuracy:
        print("Decision Tree performed better on this test set.")
    else:
        print("Both models achieved the same test accuracy.")

    print("\nTask 5 completed successfully!")


if __name__ == "__main__":
    main()
