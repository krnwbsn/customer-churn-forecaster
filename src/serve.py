import joblib
from src.data import load_data, split_data
from src.features import TenureBucket
from src.model import build_pipeline
from src.config import DATA_PATH

def run_serve():
    """
    Train the final churn prediction model on all available data
    and save both the trained pipeline and the tenure bucket transformer.
    """
    # Load and clean the full dataset
    df = load_data(DATA_PATH)

    # Split into training features and labels, discard the test portion
    X_train, _, y_train, _ = split_data(df)

    # Apply the custom tenure bucketing transformer
    tb = TenureBucket()
    X_train = tb.fit_transform(X_train)

    # Identify numeric and categorical columns
    numeric_cols     = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Build the preprocessing + classifier pipeline
    pipeline = build_pipeline(categorical_cols, numeric_cols)

    # Train the pipeline on the full training set
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline and transformer for later inference
    joblib.dump(
        {
            'pipeline': pipeline,
            'tenure_bucket': tb
        },
        'outputs/churn_model_artifacts.pkl'
    )

    # Confirm completion
    print("Model and transformer saved to outputs/churn_model_artifacts.pkl")
