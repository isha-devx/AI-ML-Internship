# Task 1 — Data Cleaning & Preprocessing

## Objective
Clean and prepare a raw dataset (Titanic passengers) so it is ready for machine learning.

## Dataset
Titanic passenger data (891 rows in the standard file) —
[source](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).
The script looks for `titanic.csv` in this folder first; if it isn't there it downloads
the file once and caches it. If the download fails, it prints the link and the filename to save it as.

## Steps performed
1. Inspected the data (`head`, `info`, `describe`, null counts).
2. **Missing values** — `Age` filled with the median, `Embarked` with the mode, `Cabin` dropped (~77% missing).
3. **Encoding** — `Sex` mapped to 0/1 (binary encoding); `Embarked` one-hot encoded (`drop_first=True`).
4. **Scaling** — `Age` and `Fare` standardised with `StandardScaler`.
5. **Outliers** — boxplots for Age and Fare, then extreme `Fare` outliers removed with the 1.5×IQR rule.
6. Saved the cleaned dataset to `outputs/titanic_cleaned.csv`.

## How to run
```bash
pip install -r ../requirements.txt
python task1_data_preprocessing.py
```
Outputs (boxplots and cleaned CSV) are written to `outputs/`.
`task1_data_preprocessing.ipynb` is the original notebook version.

## Interview Questions & Answers
1. **Types of Missing Data** → MCAR, MAR, MNAR  
2. **Handling Categorical Variables** → One-hot, Label encoding, Frequency encoding  
3. **Normalization vs Standardization** → Normalization scales to [0,1]; Standardization sets mean=0, std=1  
4. **Detect Outliers** → Boxplots, IQR, Z-score  
5. **Why Preprocessing is Important?** → Removes noise, improves accuracy  
6. **One-hot vs Label Encoding** → One-hot → multiple columns, Label → numeric codes  
7. **Handling Imbalance** → Oversampling, Undersampling, SMOTE  
8. **Does Preprocessing Affect Accuracy?** → Yes, directly improves or harms performance
