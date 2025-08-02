from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from src.inference import predict_single

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
    return predict_single(customer.dict())
