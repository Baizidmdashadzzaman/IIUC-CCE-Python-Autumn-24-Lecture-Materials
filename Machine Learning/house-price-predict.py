#pip install scikit-learn matplotlib pandas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create dataset
house_size = np.array([650, 800, 850, 900, 1200, 1500, 1600, 1800, 2000, 2300, 2500]).reshape(-1, 1)
house_price = np.array([200000, 250000, 270000, 290000, 350000, 380000, 400000, 420000, 450000, 480000, 500000])


X_train, X_test, y_train, y_test = train_test_split(house_size, house_price, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# Predict house prices
y_pred = model.predict(X_test)

# Calculate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")


# Plot the training data and the model's prediction line
plt.scatter(house_size, house_price, color='blue', label='Actual Prices')
plt.plot(house_size, model.predict(house_size), color='red', label='Predicted Prices')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.title('House Size vs Price')
plt.legend()
plt.show()

