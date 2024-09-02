import pandas as pd
from sklearn.preprocessing import RobustScaler
from Diabetes_Prediction_with_Logistic_Regression.src.utils.helpers import check_outlier, replace_with_thresholds

def data_preprocessing(df):
    """
    Preprocess the data by handling outliers and scaling features.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """
    cols = [col for col in df.columns if "Outcome" not in col]
    for col in cols:
        if check_outlier(df, col):
            df = replace_with_thresholds(df, col)
        df[col] = RobustScaler().fit_transform(df[[col]])
    return df
