import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")
 
# Features and target
X = data[["Area"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42 
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===== House Price Prediction =====\n")

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(predictions)

print("\nMean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# User prediction
print("\n----------- Predict New House Price -----------")

try:
    area = float(input("Enter House Area (sq ft): "))
    predicted_price = model.predict([[area]])
    print(f"\nEstimated House Price: ₹{predicted_price[0]:,.2f}")
except ValueError:
    print("Please enter a valid number.")

# Graph
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.title("House Area vs House Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid(True)
plt.show()