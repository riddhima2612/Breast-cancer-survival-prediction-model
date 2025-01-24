# Import necessary libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the data into features and target
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Take input from the user for new patient's features
print("Please provide the following information for the new patient:")
new_features = []
for i in range(len(data.feature_names)):
    feature_value = float(input(f"{data.feature_names[i]}: "))
    new_features.append(feature_value)

# Standardize the new features
new_features = scaler.transform([new_features])

# Predict the survival probability for the new patient
survival_probability = model.predict_proba(new_features)[0][1] * 100
print("Survival probability for the new patient:", survival_probability, "%")