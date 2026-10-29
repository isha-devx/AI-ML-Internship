# AI & ML Internship - Task 3
# House Price Prediction using Linear Regression

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Create a Sample Dataset
np.random.seed(42)

data = pd.DataFrame({
    "square_feet": np.random.randint(500, 3500, 200),
    "bedrooms": np.random.randint(1, 6, 200),
    "bathrooms": np.random.randint(1, 4, 200)
})

# Generate sample house prices
noise = np.random.randint(-20000, 20000, 200)

data["price"] = (
    data["square_feet"] * 150
    + data["bedrooms"] * 15000
    + data["bathrooms"] * 20000
    + noise
)

print("First 5 Rows of Dataset:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nSummary Statistics:")
print(data.describe())

# Step 3: Select Features and Target
X = data[["square_feet", "bedrooms", "bathrooms"]]
y = data["price"]

# Step 4: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict House Prices
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": y_pred
})

results["Predicted Price"] = results["Predicted Price"].round(2)

print("\nActual vs Predicted Prices:")
print(results.head(10))

# Step 7: Evaluate Model Performance
mae = float(mean_absolute_error(y_test, y_pred))
mse = float(mean_squared_error(y_test, y_pred))
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_test, y_pred))

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Step 8: Visualize Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# Step 9: Display Model Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

intercept = float(model.intercept_)
print("\nModel Intercept:", round(intercept, 2))

# Step 10: Predict Price for a New House
new_house = pd.DataFrame({
    "square_feet": [1800],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(new_house)
estimated_price = float(predicted_price[0])

print("\nNew House Details:")
print("Square Feet: 1800")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Estimated Price:", round(estimated_price, 2))

print("\nTask 3 Linear Regression completed successfully!")