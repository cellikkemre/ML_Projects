import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculate outlier thresholds for a given column in the dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - col_name (str): The column name to calculate the thresholds for.
    - q1 (float): Lower quantile for the threshold calculation.
    - q3 (float): Upper quantile for the threshold calculation.

    Returns:
    - tuple: Lower and upper thresholds.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    if dataframe[col_name].dtype == "int64":
        return int(low_limit), int(up_limit)
    else:
        return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Check if there are outliers in a specific column of the dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - col_name (str): The column name to check for outliers.

    Returns:
    - bool: True if there are outliers, otherwise False.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    """
    Replace outliers in a specific column with threshold values.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - variable (str): The column name to replace outliers in.

    Returns:
    - pd.DataFrame: Dataframe with outliers replaced.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe


def plot_confusion_matrix(y, y_pred):
    """
    Plot the confusion matrix for the given true and predicted values.

    Parameters:
    - y (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - None
    """
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Accuracy Score: {acc}', size=10)
    plt.show()
