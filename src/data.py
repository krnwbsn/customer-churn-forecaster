import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    """
    Load the dataset from CSV, clean and encode it for modeling.
    """
    # read raw data from CSV
    df = pd.read_csv(path)

    # convert TotalCharges to numeric, invalid parsing becomes NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # remove rows where tenure is zero
    df = df[df['tenure'] > 0].copy()

    # fill missing TotalCharges with the column mean
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    # encode target column: 'No' -> 0, 'Yes' -> 1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratified sampling.
    Returns X_train, X_test, y_train, y_test.
    """
    # separate features (X) and target (y)
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    # perform stratified train/test split
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
