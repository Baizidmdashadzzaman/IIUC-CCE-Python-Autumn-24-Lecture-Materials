#pip install scikit-learn matplotlib pandas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create dataset (hours studied and corresponding CGPA)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
cgpa = np.array([2.0, 2.3, 2.5, 2.7, 3.0, 3.2, 3.4, 3.7, 3.9, 4.0])

X_train, X_test, y_train, y_test = train_test_split(study_hours, cgpa, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# Predict CGPA
y_pred = model.predict(X_test)

# Calculate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")


# Plot the training data and the model's prediction line
plt.scatter(study_hours, cgpa, color='blue', label='Actual CGPA')
plt.plot(study_hours, model.predict(study_hours), color='red', label='Predicted CGPA')
plt.xlabel('Hours Studied')
plt.ylabel('CGPA')
plt.title('Study Hours vs CGPA')
plt.legend()
plt.show()

