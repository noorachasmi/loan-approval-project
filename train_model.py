import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create synthetic educational data
rng = np.random.default_rng(42)
number_of_applicants = 2000

annual_income = rng.integers(120000, 2000001, number_of_applicants)
loan_amount = rng.integers(25000, 1000001, number_of_applicants)
credit_score = rng.integers(300, 851, number_of_applicants)
employment_years = rng.integers(0, 31, number_of_applicants)
existing_debt = rng.integers(0, 500001, number_of_applicants)

# Make demo labels using invented rules.
# These are not real bank lending rules.
income_ok = annual_income >= loan_amount * 1.5
credit_ok = credit_score >= 650
employment_ok = employment_years >= 2
debt_ok = existing_debt <= annual_income * 0.30

demo_score = (
    income_ok.astype(int)
    + credit_ok.astype(int)
    + employment_ok.astype(int)
    + debt_ok.astype(int)
)

approved = (demo_score >= 3).astype(int)

data = pd.DataFrame({
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "existing_debt": existing_debt,
    "demo_result": approved
})

X = data.drop(columns=["demo_result"])
y = data["demo_result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Demo model evaluation")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "loan_model.joblib")
print("Saved model to loan_model.joblib")