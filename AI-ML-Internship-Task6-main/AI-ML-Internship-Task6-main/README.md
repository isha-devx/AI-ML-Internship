# AI-ML-Internship-Task6
Implementation of K-Nearest Neighbors (KNN) for classification using the Iris dataset. Includes feature normalization, experimentation with different K values, evaluation using accuracy and confusion matrix, and visualization.
# Task 6: K-Nearest Neighbors (KNN) Classification  

## 🎯 Objective  
Understand and implement KNN for classification problems.  

## 🛠 Tools Used  
- Python  
- Scikit-learn  
- Pandas  
- Matplotlib  

## 📂 Dataset  
Used the **Iris dataset** (built-in in sklearn).  

## 🚀 Steps Implemented  
1. Loaded Iris dataset.  
2. Normalized features using `StandardScaler`.  
3. Applied **KNeighborsClassifier** from sklearn.  
4. Experimented with different values of **K**.  
5. Evaluated the model using accuracy score & confusion matrix.  
6. Visualized decision boundaries for better understanding.  

## 📊 Results  
- Best accuracy achieved at **K = 5**.  
- Confusion matrix shows correct classification for most test samples.  

## 📚 Key Learnings  
- Instance-based learning  
- Role of Euclidean distance  
- K selection and its impact  

## ❓ Interview Questions  
1. **How does the KNN algorithm work?**  
   - It classifies a data point based on majority voting from nearest neighbors.  

2. **How do you choose the right K?**  
   - Using experimentation/cross-validation. Odd K helps avoid ties.  

3. **Why is normalization important in KNN?**  
   - Because distance metrics are scale-sensitive.  

4. **What is the time complexity of KNN?**  
   - Training: O(1), Prediction: O(N × D).  

5. **Pros and Cons of KNN**  
   - ✅ Simple, no training phase  
   - ❌ Slow for large datasets, sensitive to noise  

6. **Is KNN sensitive to noise?**  
   - Yes, noisy data points can reduce accuracy.  

7. **How does KNN handle multi-class problems?**  
   - By majority voting among neighbors for each class.  

8. **What’s the role of distance metrics in KNN?**  
   - Defines similarity. Common: Euclidean, Manhattan, Minkowski.  

## ▶️ How to Run  
```bash
# Clone repo
git clone https://github.com/your-username/KNN-Classification.git  

# Install libraries
pip install pandas matplotlib scikit-learn  

# Run script
python knn_classification.py
