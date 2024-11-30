#pip install scikit-learn pandas matplotlib

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()

# Create a DataFrame for better understanding
data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
data['species'] = iris['target']

# Display the first few rows
print(data.head())


X = data.iloc[:, :-1]  # Features
y = data['species']     # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")


# Plot feature importance
plt.barh(iris['feature_names'], clf.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Iris Classification')
plt.show()

