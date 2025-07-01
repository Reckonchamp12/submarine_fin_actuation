import pandas as pd
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    """
    Load data from a CSV file.
    In this NDA-safe version, this function is disabled by default.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("CSV file not found or inaccessible due to confidentiality.")
        return pd.DataFrame()

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print and return the Mean Squared Error of predictions.
    """
    mse = mean_squared_error(y_true, y_pred)
    print(f"{model_name} Mean Squared Error: {mse:.4f}")
    return mse
