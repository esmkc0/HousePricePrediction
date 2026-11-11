
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("HOUSE PRICE PREDICTION")
print("----------------------")

# Separate input features and target values
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Predict house values for the test set
y_pred = model.predict(X_test)

# Calculate model evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: ${mae * 100000:,.2f}")
print(f"RMSE: ${rmse * 100000:,.2f}")
print(f"R² Score: {r2:.4f}")

# Compare actual and predicted values
print("\nActual vs Predicted House Values:")
for actual, predicted in zip(y_test.iloc[:5], y_pred[:5]):
    print(
        f"Actual: ${actual * 100000:,.2f} | "
        f"Predicted: ${predicted * 100000:,.2f}"
    )

import matplotlib.pyplot as plt

# Create a scatter plot of actual and predicted house values
plt.figure(figsize=(9, 6))

plt.scatter(
    y_test * 100000,
    y_pred * 100000,
    alpha=0.3,
    color="steelblue",
    label="Predictions"
)

# Add a reference line for perfect predictions
min_value = min(y_test.min(), y_pred.min()) * 100000
max_value = max(y_test.max(), y_pred.max()) * 100000

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

# Add chart labels and title
plt.title("Actual vs Predicted House Values")
plt.xlabel("Actual House Value (USD)")
plt.ylabel("Predicted House Value (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the chart as an image
plt.savefig("actual_vs_predicted.png", dpi=300)

# Display the chart
plt.show()
