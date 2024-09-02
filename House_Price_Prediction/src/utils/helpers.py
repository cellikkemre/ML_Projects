import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def check_df(dataframe):
    """
    Prints basic information about the dataframe including shape, data types,
    head, tail, missing values, and quantiles.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to check.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numeric_columns = dataframe.select_dtypes(include=['number']).columns
    print(dataframe[numeric_columns].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Grabs the categorical, numerical, and categorical but cardinal column names from the dataframe.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to analyze.
    cat_th (int): Threshold for categorical columns.
    car_th (int): Threshold for categorical but cardinal columns.

    Returns:
    tuple: Categorical columns, categorical but cardinal columns, numerical columns.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, cat_but_car, num_cols


def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    """
    Calculate the lower and upper thresholds for outliers based on quantiles.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the variable.
    variable (str): The column name to analyze.
    low_quantile (float): The lower quantile threshold.
    up_quantile (float): The upper quantile threshold.

    Returns:
    tuple: Lower and upper thresholds for the variable.
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Check if the column has outliers.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the column.
    col_name (str): The column name to check for outliers.

    Returns:
    bool: True if outliers are present, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    """
    Replace outliers with threshold values.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the variable.
    variable (str): The column name for which to replace outliers.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    """
    Generate a table of missing values and their ratios.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to check.
    na_name (bool): Whether to return the names of columns with missing values.

    Returns:
    pd.Index: Columns with missing values if na_name is True.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    """
    Quickly imputes missing values based on the method provided.

    Parameters:
    data (pd.DataFrame): The dataframe to impute.
    num_method (str): The method for numeric variables ('mean' or 'median').
    cat_length (int): The threshold for determining whether a categorical variable will be imputed with mode.
    target (str): The target variable name, used to retain the original target.

    Returns:
    pd.DataFrame: The dataframe with imputed values.
    """
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(f"Imputation method is '{num_method.upper()}' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


def rare_analyser(dataframe, target, cat_cols):
    """
    Analyze rare categories in categorical columns.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to analyze.
    target (str): The target variable.
    cat_cols (list): List of categorical columns to analyze.
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    """
    Encode rare categories as 'Rare'.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to encode.
    rare_perc (float): The threshold below which categories are considered rare.

    Returns:
    pd.DataFrame: The dataframe with rare categories encoded.
    """
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def label_encoder(dataframe, binary_col):
    """
    Apply Label Encoding to binary categorical columns.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to encode.
    binary_col (str): The column name to encode.

    Returns:
    pd.DataFrame: The dataframe with the column label encoded.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Apply One-Hot Encoding to categorical columns.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to encode.
    categorical_cols (list): List of categorical columns to encode.
    drop_first (bool): Whether to drop the first category.

    Returns:
    pd.DataFrame: The dataframe with one-hot encoded columns.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def plot_importance(model, features, num, save=False):
    """
    Plot the importance of features in a trained model.

    Parameters:
    model: The trained model with feature importances.
    features (pd.DataFrame): The features dataframe.
    num (int): Number of features to plot.
    save (bool): Whether to save the plot.
    """
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")