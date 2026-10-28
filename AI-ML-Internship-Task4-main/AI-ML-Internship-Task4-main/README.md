# AI-ML-Internship-Task4
This project demonstrates binary classification using Logistic Regression on the Breast Cancer dataset. It includes data preprocessing, model training, evaluation with confusion matrix, precision, recall, ROC-AUC, and visualization of ROC curve &amp; sigmoid function for better understanding.

# Logistic Regression - Binary Classification

## 📌 Task Objective
Build a **binary classifier** using Logistic Regression on the Breast Cancer dataset.

## ⚙️ Tools Used
- Python
- Scikit-learn
- Pandas
- Matplotlib

## 🚀 Steps Performed
1. Loaded Breast Cancer dataset from Scikit-learn.
2. Split data into train and test sets.
3. Standardized features using StandardScaler.
4. Built a Logistic Regression model.
5. Evaluated model using:
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC Score
6. Visualized **ROC Curve** and **Sigmoid Function**.

## 📊 Results
- Achieved high accuracy and ROC-AUC score.
- Showed how logistic regression works for binary classification.
- Explained sigmoid curve behavior.

## 📂 Files
- `task4_logistic_regression.py` → Main code
- `README.md` → Documentation

---
✅ **Task Completed**

✅ Interview Questions & Answers

1. How does logistic regression differ from linear regression?

Linear Regression predicts continuous values (e.g., house price).

Logistic Regression predicts probabilities for classification (e.g., cancer vs. no cancer).

Logistic uses sigmoid function to map predictions to the range [0,1].




2. What is the sigmoid function?

The sigmoid function is:


\sigma(z) = \frac{1}{1 + e^{-z}}

Used in logistic regression to model probability of belonging to a class.



3. What is precision vs recall?

Precision: Out of all predicted positives, how many were correct.


Precision = \frac{TP}{TP + FP}

Recall = \frac{TP}{TP + FN}

Recall = ability to find all positives.




4. What is the ROC-AUC curve?

ROC (Receiver Operating Characteristic) curve plots TPR vs FPR at different thresholds.

AUC (Area Under Curve) measures classifier performance.

AUC = 1 → Perfect classifier, AUC = 0.5 → Random guess.



5. What is the confusion matrix?

A 2×2 table showing classification results:


	Predicted Positive	Predicted Negative

Actual Positive	True Positive (TP)	False Negative (FN)
Actual Negative	False Positive (FP)	True Negative (TN)


Helps calculate accuracy, precision, recall, and F1 score.



6. What happens if classes are imbalanced?

Model may get biased toward the majority class.

Metrics like accuracy become misleading.

Solution: Use resampling, SMOTE, class weights, ROC-AUC, precision-recall curve.





7. How do you choose the threshold?

Default threshold = 0.5.

Can be adjusted based on:

High Recall needed → lower threshold.

High Precision needed → higher threshold.


Chosen by evaluating ROC curve or Precision-Recall curve.



8. Can logistic regression be used for multi-class problems?

Yes ✅

Using One-vs-Rest (OvR) or Multinomial Logistic Regression (Softmax).

Scikit-learn’s LogisticRegression(multi_class="multinomial") supports this.
