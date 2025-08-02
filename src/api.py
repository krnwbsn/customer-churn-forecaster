from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# load saved model pipeline and tenure bucket transformer
artifacts = joblib.load('outputs/churn_model_artifacts.pkl')
pipeline = artifacts['pipeline']
tenure_bucket = artifacts['tenure_bucket']

# create FastAPI app
app = FastAPI(title="Customer Churn Forecast API")

# define input schema for a customer
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1) # 0 or 1
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=1) # must be >= 1
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: Customer):
    """
    receive a JSON payload for one customer,
    compute churn probability,
    and return the probability plus a simple conclusion
    """
    # convert the pydantic model to a DataFrame
    df = pd.DataFrame([customer.dict()])

    # apply the tenure bucket transformer
    X = tenure_bucket.transform(df)

    # predict probability of churn (index 1 is P(churn=Yes))
    proba = pipeline.predict_proba(X)[:, 1][0]
    proba_rounded = round(float(proba), 3)

    # determine conclusion based on a 0.5 threshold
    if proba_rounded >= 0.5:
        conclusion = "Customer is likely to churn"
    else:
        conclusion = "Customer is unlikely to churn"

    # return both probability and conclusion
    return {
        "churn_probability": proba_rounded,
        "conclusion": conclusion
    }
