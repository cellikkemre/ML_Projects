import pandas as pd


def load_and_preview_data(filepath):
    """
    Load and preview the dataset.

    Parameters:
    - filepath (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(filepath)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.width', 500)
    return df
