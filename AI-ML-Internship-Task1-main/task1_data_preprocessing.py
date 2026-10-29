# AI & ML Internship - Task 1
# Data Cleaning & Preprocessing

# Step 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include="all"))

# Step 2: Handle Missing Values
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because it contains many missing values
df = df.drop(columns=["Cabin"])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Step 3: Encode Categorical Features
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

print("\nEncoded Dataset Sample:")
print(df.head())

# Step 4: Standardize Numerical Features
scaler = StandardScaler()
num_cols = ["Age", "Fare"]

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nScaled Numerical Features:")
print(df[num_cols].head())

# Step 5: Outlier Detection and Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["Age"])
plt.title("Boxplot of Age")

plt.subplot(1, 2, 2)
sns.boxplot(y=df["Fare"])
plt.title("Boxplot of Fare")

plt.tight_layout()
plt.show()

# Step 6: Remove Extreme Fare Outliers
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

df = df[df["Fare"] <= upper_limit].copy()

print("\nDataset Shape After Outlier Removal:")
print(df.shape)

print("\nFinal Missing Values Check:")
print(df.isnull().sum())

print("\nTask 1 completed successfully!")
