# AI & ML Internship - Task 1
## Data Cleaning & Preprocessing

### 📌 Objective
Learn how to clean and prepare raw data for ML.

### 🛠 Tools Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

### 📂 Dataset
Titanic Dataset: [Download Here](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

### 🚀 Steps Performed
1. **Imported Dataset** and explored basic info (nulls, data types, summary stats).  
2. **Handled Missing Values**  
   - Age → filled with median  
   - Embarked → filled with mode  
   - Cabin → dropped (too many missing values)  
3. **Encoded Categorical Features**  
   - Sex → Label Encoding (Male=0, Female=1)  
   - Embarked → One-hot encoding  
4. **Normalized Numerical Features**  
   - Standardized Age & Fare using `StandardScaler`.  
5. **Detected Outliers**  
   - Used boxplots for Age and Fare.  
   - Removed extreme outliers in Fare using IQR method.  

---

### 📊 Visualizations
- Boxplot of Age  
- Boxplot of Fare  

---

### 📌 Interview Questions & Answers
1. **Types of Missing Data** → MCAR, MAR, MNAR  
2. **Handling Categorical Variables** → One-hot, Label encoding, Frequency encoding  
3. **Normalization vs Standardization** → Normalization scales to [0,1]; Standardization sets mean=0, std=1  
4. **Detect Outliers** → Boxplots, IQR, Z-score  
5. **Why Preprocessing is Important?** → Removes noise, improves accuracy  
6. **One-hot vs Label Encoding** → One-hot → multiple columns, Label → numeric codes  
7. **Handling Imbalance** → Oversampling, Undersampling, SMOTE  
8. **Does Preprocessing Affect Accuracy?** → Yes, directly improves or harms performance  

---

### 📂 Repository Structure
