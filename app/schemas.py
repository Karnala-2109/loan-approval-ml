from pydantic import BaseModel


class LoanApplication(BaseModel):
    age: int
    income: float
    credit_score: int
    loan_amount: float
    employment_years: int
    existing_loans: int