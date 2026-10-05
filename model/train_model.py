import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib


# Load dataset
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


# Target variable
y = data["loan_approved"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LogisticRegression(
    max_iter=1000
)


# Train model
model.fit(
    X_train,
    y_train
)


# Make predictions
predictions = model.predict(X_test)


# Model evaluation
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# Display evaluation results
print("\n========== MODEL EVALUATION ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)


# Confusion Matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# Classification Report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save trained model
joblib.dump(
    model,
    "model/loan_model.pkl"
)


print("\nModel saved successfully!")