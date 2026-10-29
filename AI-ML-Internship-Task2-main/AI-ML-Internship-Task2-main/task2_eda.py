# AI & ML Internship - Task 2
# Exploratory Data Analysis (EDA)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Dataset Information
print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Histograms of Numerical Features
numeric_columns = df.select_dtypes(include="number").columns

df[numeric_columns].hist(bins=20, figsize=(12, 10), edgecolor="black")

plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# Step 5: Age Distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# Step 6: Boxplot of Age
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Age")
plt.title("Age Distribution and Outliers")
plt.tight_layout()
plt.show()

# Step 7: Fare by Passenger Class
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.title("Fare Distribution by Passenger Class")
plt.tight_layout()
plt.show()

# Step 8: Survival Analysis
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Count")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.tight_layout()
plt.show()

# Step 9: Pairplot
pairplot_data = df[["Age", "Fare", "Pclass", "Survived"]].dropna()

sns.pairplot(pairplot_data, hue="Survived", diag_kind="hist")
plt.show()

# Step 10: Correlation Matrix
correlation_data = df.select_dtypes(include="number")
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

# Step 11: Skewness Analysis
print("\nSkewness of Age and Fare:")
print(df[["Age", "Fare"]].skew())

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Fare", bins=30, kde=True)
plt.title("Fare Distribution and Skewness")
plt.tight_layout()
plt.show()

# Step 12: Basic Observations
print("\nBasic Observations:")
print("- The dataset contains information about Titanic passengers.")
print("- Passenger class 3 contains the most passengers.")
print("- Age contains missing values.")
print("- Cabin contains many missing values.")
print("- Fare is positively skewed, with some high-fare outliers.")
print("- The survival column contains two classes: 0 and 1.")
print("- Correlation analysis shows relationships between numerical features.")

print("\nTask 2 EDA completed successfully!")
