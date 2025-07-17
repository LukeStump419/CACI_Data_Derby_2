import pandas as pd

def normalize_columns(df):
    """Standardize column names: lowercase, strip, replace spaces with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def load_train_data(path="data/train.csv", return_X_y=True):
    """Loads and cleans the training dataset."""
    df = pd.read_csv(path)
    df = normalize_columns(df)

    if return_X_y:
        X = df.drop(columns=["id", "quality"])
        y = df["quality"]
        return X, y
    else:
        return df

def load_test_data(path="data/test.csv"):
    """Loads and cleans the test dataset."""
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df