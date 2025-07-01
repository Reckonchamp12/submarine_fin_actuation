import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load data from a CSV file.
    In this NDA-safe version, will warn if file is missing.
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
    Calculate and print MSE of the model.
    """
    mse = mean_squared_error(y_true, y_pred)
    print(f"{model_name} Mean Squared Error: {mse:.4f}")
    return mse

def plot_correlation_heatmap(df, title="Feature Correlation Heatmap"):
    """
    Plot a correlation heatmap of the DataFrame features.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, columns, bins=30):
    """
    Plot histogram distributions of selected features.
    """
    df[columns].hist(bins=bins, figsize=(12, 8), grid=False)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

def plot_prediction_scatter(y_true, y_pred, model_name="Model"):
    """
    Scatter plot of actual vs predicted values.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

