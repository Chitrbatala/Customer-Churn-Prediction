from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_data():
    """
    Load the raw customer churn dataset.
    """
    df = pd.read_csv(DATA_PATH)

    return df


def clean_data(df):
    """
    Perform basic data cleaning.
    """

    # Convert TotalCharges from string to numeric.
    # Invalid/blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Customers with tenure = 0 have no accumulated charges.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def prepare_features(df):
    """
    Separate features and target variable.
    """

    # customerID is an identifier, not a predictive feature.
    X = df.drop(
        columns=["customerID", "Churn"]
    )

    # Target variable
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


if __name__ == "__main__":

    df = load_data()

    df = clean_data(df)

    X, y = prepare_features(df)

    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nTarget distribution:")
    print(y.value_counts())