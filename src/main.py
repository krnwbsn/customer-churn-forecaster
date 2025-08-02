import pandas as pd
from src.config import DATA_PATH
from src.utils import load_data
from src.plots import (
    plot_missing_matrix,
    plot_gender_churn,
    plot_contract_distribution,
    plot_payment_method_distribution,
    plot_payment_method_churn,
    plot_internet_gender_churn,
    plot_binary_churn,
    plot_monthly_total_charges,
    plot_correlation
)
from src.data import load_data as load_ml_data, split_data
from src.features import TenureBucket
from src.model import build_pipeline
from src.evaluate import cross_validate, train_final, evaluate
from src.tuning import tune_pipeline
from src.interpret import explain_model
from src.serve import run_serve

if __name__ == '__main__':
    # EDA and Visualization setup
    # load raw data for plotting (keeps churn as 'Yes'/'No')
    df_raw = load_data(DATA_PATH)
    # convert TotalCharges to numeric, invalid parsing becomes NaN
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    # drop rows where tenure is zero
    df_raw = df_raw[df_raw['tenure'] > 0].copy()
    # fill any missing TotalCharges with the mean
    mean_tc = df_raw['TotalCharges'].mean()
    df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(mean_tc)
    # map SeniorCitizen from 0/1 to 'No'/'Yes' for plotting
    df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # generate all EDA plots and save to outputs/
    plot_missing_matrix(df_raw)
    plot_gender_churn(df_raw)
    plot_contract_distribution(df_raw)
    plot_payment_method_distribution(df_raw)
    plot_payment_method_churn(df_raw)
    plot_internet_gender_churn(df_raw)
    plot_binary_churn(
        df_raw,
        ['Partner', 'Dependents', 'TechSupport',
         'OnlineSecurity', 'PaperlessBilling', 'PhoneService', 'SeniorCitizen']
    )
    plot_monthly_total_charges(df_raw)
    plot_correlation(df_raw)

    # Modeling pipeline (prepare data for ML)
    # load and clean data, map churn to 0/1
    df = load_ml_data(DATA_PATH)
    # split into train and test sets with stratification
    X_train, X_test, y_train, y_test = split_data(df)

    # apply custom transformer to bucket tenure into categories
    tb = TenureBucket()
    X_train = tb.fit_transform(X_train)
    X_test  = tb.transform(X_test)

    # define numeric and categorical feature columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # build the sklearn pipeline (preprocessing + classifier)
    pipeline = build_pipeline(categorical_cols, numeric_cols)

    # Cross validation and final evaluation
    # run stratified cross validation and show CV ROC-AUC
    cross_validate(pipeline, X_train, y_train)

    # train final model on the full training set
    model = train_final(pipeline, X_train, y_train)
    # evaluate on the hold-out test set
    evaluate(model, X_test, y_test)

    # Hyperparameter tuning
    # parameter grid for LogisticRegression
    param_dist = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear']
    }
    print("Tuning hyperparameters...")
    # run randomized search to optimize ROC-AUC
    best_pipeline = tune_pipeline(pipeline, param_dist, X_train, y_train)

    # retrain best pipeline on full training set
    best_pipeline.fit(X_train, y_train)
    # evaluate tuned model on test set
    evaluate(best_pipeline, X_test, y_test)

    # Model interpretation
    # generate SHAP summary plots for the tuned model
    explain_model(best_pipeline, X_train)

    # Save artifacts and run batch scoring
    # save the final model and transformer for inference
    run_serve()
