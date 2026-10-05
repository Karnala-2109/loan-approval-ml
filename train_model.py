import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("data/loan_data.csv")

print("Dataset loaded successfully!")
print(data)

# Input features
X = data[
    [
        "age",
        "income",
        "credit_score",
        "loan_amount",
        "employment_years",
        "existing_loans"
    ]
]

# Target
y = data["loan_approved"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "model/loan_model.pkl")

print("Model saved successfully!")