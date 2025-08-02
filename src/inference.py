import joblib
import pandas as pd
from pathlib import Path

# Load once at import time
ARTIFACT_PATH = Path(__file__).parents[1] / "outputs" / "churn_model_artifacts.pkl"
_artifacts = joblib.load(ARTIFACT_PATH)
_pipeline = _artifacts["pipeline"]
_tenure_bucket = _artifacts["tenure_bucket"]

def predict_single(sample: dict) -> dict:
    """
    sample: a dict of raw feature values
    returns: { churn_probability: float, conclusion: str }
    """
    # raw -> DataFrame
    df = pd.DataFrame([sample])

    # only transformed features -> array or sparse matrix
    X = _tenure_bucket.transform(df)

    # feed that into classifier
    proba = _pipeline.predict_proba(X)[:, 1][0]
    proba = round(float(proba), 3)

    # human-friendly conclusion
    conclusion = (
        "Customer is likely to churn"
        if proba >= 0.5
        else "Customer is unlikely to churn"
    )

    return {"churn_probability": proba, "conclusion": conclusion}

