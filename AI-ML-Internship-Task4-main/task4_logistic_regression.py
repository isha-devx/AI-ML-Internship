
# Task 4: Logistic Regression
# AI/ML Internship

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)


def main():
    # 2. Load the dataset
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(dataset.target, name="target")
    class_names = dataset.target_names

    print("Dataset loaded successfully!")
    print("Dataset shape:", X.shape)

    print("\nFirst five rows:")
    print(X.head())

    print("\nTarget classes:")
    print(class_names)

    print("\nClass distribution:")
    print(
        y.map({0: "malignant", 1: "benign"}).value_counts()
    )

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # 4. Standardize the features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train the Logistic Regression model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    print("\nLogistic Regression model trained successfully!")

    # 6. Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 7. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0,
        )
    )

    # 8. Display the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\nConfusion Matrix:")
    print(cm)

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    matrix_display.plot(cmap="Blues")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 9. Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 10. Plot sigmoid function
    z = np.linspace(-10, 10, 500)
    sigmoid = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(8, 6))
    plt.plot(z, sigmoid, color="darkgreen", linewidth=2)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.title("Sigmoid Function")
    plt.xlabel("z")
    plt.ylabel("σ(z)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 11. Predict one test sample
    sample = X_test_scaled[0].reshape(1, -1)
    prediction = int(model.predict(sample)[0])
    actual = int(y_test.iloc[0])

    print("\nSample Prediction")
    print("-----------------")
    print("Predicted class:", class_names[prediction])
    print("Actual class:", class_names[actual])

    print("\nTask 4 Logistic Regression completed successfully!")


if __name__ == "__main__":
    main()
