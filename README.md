# AI & Machine Learning Internship Projects

Six hands-on ML tasks plus a capstone-style disease outbreak prediction project, built with Python and scikit-learn.
Every script runs end to end, saves its figures to an `outputs/` folder, and reports honest evaluation results.

## Projects

| # | Project | Dataset | Key result |
|---|---|---|---|
| 1 | [Data Preprocessing](task1_data_preprocessing) | Titanic | Missing values, encoding, scaling, IQR outlier removal |
| 2 | [Exploratory Data Analysis](task2_eda) | Titanic | Distributions, correlations, survival by class & sex |
| 3 | [Linear Regression](task3_linear_regression) | Diabetes (442 patients) | R² 0.23 (simple) → 0.45 (multiple); multicollinearity found via VIF |
| 4 | [Logistic Regression](task4_logistic_regression) | Breast Cancer Wisconsin | 98.25% accuracy, ROC-AUC 0.995 |
| 5 | [Decision Tree & Random Forest](task5_decision_tree_random_forest) | UCI Heart Disease (297) | RF 86.7% vs tree 80.0% test; 82.6% vs 80.1% in 5-fold CV |
| 6 | [K-Nearest Neighbors](task6_knn_classification) | Iris | 91–93% accuracy for K = 3, 5, 7 |
| ★ | [Disease Outbreak Prediction](disease_outbreak_prediction) | **Simulated** weekly surveillance data | Time-based split, 3 models + baselines, ROC-AUC ≈ 0.97 |

> The outbreak project uses a clearly-labelled **simulated** dataset (no free real outbreak dataset exists in one clean file).
> Its pipeline accepts real data in the same format — see its README.

## Quick start
```bash
git clone https://github.com/isha-devx/AI-ML-Internship.git
cd AI-ML-Internship
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py            # smoke-tests every project (no plot windows)
# or run one project:
cd task4_logistic_regression && python task4_logistic_regression.py
```
Tasks 1–2 download the Titanic CSV on first run and cache it next to the script.

## Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## Repository layout
```
AI-ML-Internship/
├── requirements.txt
├── run_all.py                      # runs every script, prints PASS/FAIL
├── AI-ML-Internship-Project-Report.pdf   # full project report
├── task1_data_preprocessing/
├── task2_eda/
├── task3_linear_regression/
├── task4_logistic_regression/
├── task5_decision_tree_random_forest/
├── task6_knn_classification/
└── disease_outbreak_prediction/    # data generator, training pipeline, predictor, saved model
```

## Author
**Isha Lohani** — [@isha-devx](https://github.com/isha-devx)

*Built for learning and internship practice. Not medical advice.*
