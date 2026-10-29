# Task 4: Logistic Regression

## Objective
Build a Logistic Regression classification model to predict whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin dataset.

## Tools and Libraries
- Python
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn

## Project Workflow
1. Load the Breast Cancer Wisconsin dataset.
2. Explore the dataset and examine class distribution.
3. Split the data into training and testing sets.
4. Standardize features using `StandardScaler`.
5. Train a Logistic Regression model.
6. Evaluate performance using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
7. Visualize the ROC curve, confusion matrix, and sigmoid function.
8. Test the model with a sample prediction.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 98.25% |
| ROC-AUC Score | 0.9954 |
| Training Samples | 455 |
| Testing Samples | 114 |

### Confusion Matrix

```text
[[41  1]
 [ 1 71]]
```

The model correctly classified 112 out of 114 test samples.

## Key Concepts

- **Logistic Regression:** A supervised learning algorithm used for classification.
- **Sigmoid Function:** Converts a model's output into a value between 0 and 1.
- **StandardScaler:** Standardizes features to improve model training.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive cases are identified.
- **ROC-AUC:** Measures how effectively a model distinguishes between classes.
- **Confusion Matrix:** Summarizes correct and incorrect classifications.

## Interview Questions

**1. Why is Logistic Regression used for classification?**

It estimates the probability of a class and can classify observations using a decision threshold.

**2. What is the sigmoid function?**

It maps a real-valued input to a probability-like value between 0 and 1.

**3. Why do we use feature scaling?**

Scaling puts features on comparable ranges and can help optimization converge more effectively.

**4. What is the difference between precision and recall?**

Precision measures the reliability of positive predictions, while recall measures how many actual positive cases are detected.

**5. What does ROC-AUC represent?**

It measures the model's ability to distinguish between the two classes across different classification thresholds.

**6. What is a confusion matrix?**

It summarizes true positives, true negatives, false positives, and false negatives.

**7. Why is the train-test split important?**

It evaluates how well a model generalizes to data it did not train on.

**8. Why might accuracy be misleading on imbalanced data?**

A model can achieve high accuracy by favoring the majority class while performing poorly on the minority class.

## How to Run

From the repository root, activate your virtual environment and run:

```powershell
python ".\AI-ML-Internship-Task4-main\AI-ML-Internship-Task4-main\task4_logistic_regression.py"
```

## Conclusion

The Logistic Regression model achieved 98.25% accuracy and a ROC-AUC score of 0.9954 on the test set. The project demonstrates data preprocessing, feature scaling, classification, model evaluation, and visualization using Python and Scikit-learn.
