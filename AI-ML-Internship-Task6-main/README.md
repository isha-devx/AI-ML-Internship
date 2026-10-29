
# AI/ML Internship — Task 6: KNN Classification

## Objective
Implement the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers and evaluate its performance using different values of K.

## Dataset
- **Iris dataset**, loaded directly from scikit-learn.
- 150 samples, 4 numerical features, and 3 flower classes.

## Tools and Libraries
- Python
- Scikit-learn
- Matplotlib

## Implementation
1. Loaded the Iris dataset.
2. Split the dataset into training and testing sets.
3. Standardized the features using `StandardScaler`, fitted only on the training data.
4. Trained KNN classifiers using K = 3, 5, and 7.
5. Compared the models using accuracy.
6. Evaluated the final K=5 model using a confusion matrix.

## Results

| K | Test Accuracy |
|---|---:|
| 3 | 91.11% |
| 5 | 91.11% |
| 7 | 93.33% |

The K=7 model achieved the highest test accuracy in this experiment. The final model and confusion matrix use K=5 for comparison.

## Key Learnings
- How KNN classifies samples using nearby data points.
- Why feature scaling matters for distance-based algorithms.
- How the choice of K affects classification performance.
- How to evaluate a classifier using accuracy and a confusion matrix.

## How to Run

Install the required libraries:

```bash
pip install scikit-learn matplotlib
```

Run the script from the project folder:

```bash
python knn_classification.py
```
