import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class TenureBucket(BaseEstimator, TransformerMixin):
    """
    Divide tenure into buckets (0-6, 7-12, 13-24, 25-48, 49-72).
    """
    def __init__(self, bins=[0, 6, 12, 24, 48, 72]):
        # store the edges for tenure buckets
        self.bins = bins

    def fit(self, X, y=None):
        # no fitting needed, return self
        return self

    def transform(self, X):
        # work on a copy to avoid modifying the original DataFrame
        X = X.copy()
        # assign each tenure value to a bucket label
        X['TenureBucket'] = pd.cut(
            X['tenure'],
            bins=self.bins,
            labels=[
                f"{self.bins[i]}-{self.bins[i+1]}"
                for i in range(len(self.bins) - 1)
            ],
            include_lowest=True
        )
        # return the transformed DataFrame
        return X

def create_features(df: pd.DataFrame):
    """
    Take a cleaned DataFrame (with numeric TotalCharges and binary Churn),
    apply tenure bucketing, one-hot encode categoricals,
    and return (X, y) for modeling.
    """
    # pull out the target
    y = df["Churn"].copy()
    X = df.drop(columns=["customerID", "Churn"])
    
    # tenure → bucket
    X = TenureBucket().fit_transform(X)
    
    # pick feature lists
    #    (adjust these to match what was used at training time)
    cat_cols = [
        "gender", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "TenureBucket"
    ]
    
    # build a ColumnTransformer
    preproc = ColumnTransformer([
        ("onehot",
         OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         cat_cols),
    ], remainder="passthrough")
    
    # fit & transform
    X_trans = preproc.fit_transform(X)

    # wrap back into a DataFrame
    feature_names = preproc.get_feature_names_out()
    X_final = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    return X_final, y