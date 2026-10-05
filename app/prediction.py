from fastapi import APIRouter
import joblib
import pandas as pd

from app.schemas import LoanApplication

router = APIRouter(prefix="/api", tags=["Prediction"])

# Load trained ML model
model = joblib.load("model/loan_model.pkl")


@router.post("/predict")
def predict_loan(application: LoanApplication):

    data = pd.DataFrame([{
        "age": application.age,
        "income": application.income,
        "credit_score": application.credit_score,
        "loan_amount": application.loan_amount,
        "employment_years": application.employment_years,
        "existing_loans": application.existing_loans
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return {
        "result": result,
        "approval_probability": round(float(probability) * 100, 2)
    }