from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "loan_model.joblib"
)

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            values = {
                "annual_income": float(request.form["annual_income"]),
                "loan_amount": float(request.form["loan_amount"]),
                "credit_score": int(request.form["credit_score"]),
                "employment_years": int(
                    request.form["employment_years"]
                ),
                "existing_debt": float(request.form["existing_debt"])
            }

            if (
                values["annual_income"] <= 0
                or values["loan_amount"] <= 0
                or values["existing_debt"] < 0
                or not 300 <= values["credit_score"] <= 850
                or not 0 <= values["employment_years"] <= 60
            ):
                raise ValueError("Please enter valid values.")

            applicant = pd.DataFrame([values])

            prediction = int(model.predict(applicant)[0])

            if prediction == 1:
                result = "Demo prediction: APPROVAL"
            else:
                result = "Demo prediction: REJECTION"

        except (ValueError, TypeError, KeyError):
            error = (
                "Please check all fields and enter valid numeric values."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)