
import pandas as pd


def load_data(filepath):
    """
    Load the dataset from a specified filepath.

    Parameters:
        filepath (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return pd.read_csv(filepath)


load_data("Baseball_Salary_Prediction/data/raw/hitters.csv")