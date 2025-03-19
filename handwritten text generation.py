import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (replace this with a real dataset)
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "monthly_usage": [10, 50, 5, 70, 20],
    "customer_age": [25, 40, 22, 35, 30],
    "subscription_length": [12, 24, 6, 36, 18],
    "churn": [1, 0, 1, 0, 1]  # 1 = Churned, 0 = Retained
}

df = pd.DataFrame(data)

# Features and target variable
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Interactive loop for real-time predictions
while True:
    try:
        usage = float(input("Enter monthly usage: "))
        age = int(input("Enter customer age: "))
        subscription = int(input("Enter subscription length (months): "))
        new_X = np.array([[usage, age, subscription]])
        predicted_churn = classifier.predict(new_X)
        print(f"Predicted Churn: {'Yes' if predicted_churn[0] == 1 else 'No'}")
    except ValueError:
        print("Invalid input. Please enter numerical values.")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
