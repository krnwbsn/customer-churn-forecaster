from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

def build_pipeline(categorical_cols, numeric_cols):
    """
    Create a scikit-learn Pipeline that
    - applies preprocessing to numeric and categorical features
    - fits a logistic regression classifier
    """
    # pipeline for numeric features: scale values to zero mean and unit variance
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    # pipeline for categorical features: one-hot encode unseen categories safely
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    # combine numeric and categorical pipelines into a single transformer
    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols), # apply scaler to numeric columns
        ('cat', cat_pipe, categorical_cols) # apply one-hot to categorical columns
    ])

    # final pipeline: preprocessing followed by logistic regression
    pipeline = Pipeline([
        ('preproc', preproc),
        ('clf', LogisticRegression(
            solver='lbfgs', # optimization algorithm
            max_iter=1000, # increase iteration limit for convergence
            class_weight='balanced', # handle class imbalance automatically
            random_state=42 # for reproducible results
        ))
    ])

    return pipeline
