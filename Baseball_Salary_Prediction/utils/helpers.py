import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# General DataFrame Check Function
def check_df(dataframe, head=5):
    """
    Display general information about the dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to inspect.
        head (int): Number of rows to display from the head and tail.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Function to Grab Column Names
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Get the names of categorical, numerical, and categorical but cardinal variables in a dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to inspect.
        cat_th (int): Threshold for numerical but categorical variables.
        car_th (int): Threshold for categorical but cardinal variables.

    Returns:
        cat_cols (list): List of categorical columns.
        num_cols (list): List of numerical columns.
        cat_but_car (list): List of categorical but cardinal columns.
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

    return cat_cols, num_cols, cat_but_car


# One-Hot Encoding Function
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Apply one-hot encoding to the specified categorical columns in the dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to encode.
        categorical_cols (list): List of categorical columns to encode.
        drop_first (bool): Whether to drop the first level of the encoded columns.

    Returns:
        pd.DataFrame: The encoded dataframe.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Categorical Summary Function
def cat_summary(dataframe, col_name, plot=False):
    """
    Display summary statistics for a categorical variable.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the categorical column.
        col_name (str): The name of the categorical column.
        plot (bool): Whether to plot the distribution.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# Numerical Summary Function
def num_summary(dataframe, numerical_col, plot=False):
    """
    Display summary statistics for a numerical variable.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the numerical column.
        numerical_col (str): The name of the numerical column.
        plot (bool): Whether to plot the distribution.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# Target Variable Analysis with Categorical Features
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Display the mean of the target variable for each category of a categorical feature.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the target and categorical columns.
        target (str): The name of the target variable.
        categorical_col (str): The name of the categorical column.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


# Correlation Analysis Function
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    Identify and optionally plot highly correlated features.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to analyze.
        plot (bool): Whether to plot the correlation matrix.
        corr_th (float): Correlation threshold above which features are considered highly correlated.

    Returns:
        list: List of highly correlated column names.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


# Outlier Detection and Handling Functions
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper thresholds for detecting outliers.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the column.
        col_name (str): The name of the column.
        q1 (float): The first quartile value.
        q3 (float): The third quartile value.

    Returns:
        tuple: The lower and upper thresholds.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Check if there are any outliers in the specified column.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the column.
        col_name (str): The name of the column.

    Returns:
        bool: True if there are outliers, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    """
    Replace outliers in the specified column with the threshold values.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the column.
        variable (str): The name of the column.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Missing Values Analysis Function
def missing_values_table(dataframe, na_name=False):
    """
    Display the missing values in the dataframe and their proportions.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to analyze.
        na_name (bool): Whether to return the names of columns with missing values.

    Returns:
        list: List of column names with missing values (if na_name=True).
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


# Feature Scaling Function
def scale_features(dataframe, num_cols):
    """
    Apply standard scaling to the specified numerical columns.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the columns.
        num_cols (list): List of numerical columns to scale.

    Returns:
        pd.DataFrame: The dataframe with scaled features.
    """
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe


# Feature Engineering Function
def create_new_features(dataframe):
    """
    Create new features in the dataframe based on existing features.

    Parameters:
        dataframe (pd.DataFrame): The dataframe to modify.

    Returns:
        pd.DataFrame: The dataframe with new features added.
    """
    dataframe['NEW_Hits'] = dataframe['Hits'] / dataframe['CHits'] + dataframe['Hits']
    dataframe['NEW_RBI'] = dataframe['RBI'] / dataframe['CRBI']
    dataframe['NEW_Walks'] = dataframe['Walks'] / dataframe['CWalks']
    dataframe['NEW_PutOuts'] = dataframe['PutOuts'] * dataframe['Years']
    dataframe["Hits_Success"] = (dataframe["Hits"] / dataframe["AtBat"]) * 100
    dataframe["NEW_CRBI*CATBAT"] = dataframe['CRBI'] * dataframe['CAtBat']
    # Add all the other feature engineering steps similarly
    return dataframe


# Base Model Training Function
def train_base_models(X, y):
    """
    Train and evaluate base models.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    """
    models = [
        ('RF', RandomForestRegressor()),
        ('GBM', GradientBoostingRegressor()),
        ('LightGBM', LGBMRegressor()),
        ('CatBoost', CatBoostRegressor(verbose=False))
    ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


# Hyperparameter Optimization Function
def hyperparameter_optimization(X, y, regressors):
    """
    Perform hyperparameter optimization for a list of models.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        regressors (list): List of tuples containing model name, model object, and hyperparameter grid.

    Returns:
        dict: Dictionary containing the best models and their optimized parameters.
    """
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

    return best_models






# Validation Curve Plotting Function
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Plot validation curves to analyze model complexity.

    Parameters:
        model: The model object to validate.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        param_name (str): The hyperparameter name to validate.
        param_range (list): The range of values to validate.
        scoring (str): Scoring metric for validation.
        cv (int): Number of cross-validation folds.
    """
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
