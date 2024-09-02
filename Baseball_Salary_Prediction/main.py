from Baseball_Salary_Prediction.utils.helpers import (
    check_df, grab_col_names, one_hot_encoder, cat_summary, num_summary,
    target_summary_with_cat, high_correlated_cols, check_outlier,
    replace_with_thresholds, missing_values_table, scale_features,
    create_new_features, train_base_models, hyperparameter_optimization,
     val_curve_params
)
from Baseball_Salary_Prediction.src.data_loading import load_data


def main():
    # Load the dataset
    df = load_data("Baseball_Salary_Prediction/data/raw/hitters.csv")

    # Initial data inspection
    check_df(df)

    # Grab column names
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Categorical and numerical summaries
    for col in cat_cols:
        cat_summary(df, col, plot=True)

    for col in num_cols:
        num_summary(df, col, plot=True)

    # Target summary with categorical variables
    for col in cat_cols:
        target_summary_with_cat(df, "Salary", col)

    # Correlation analysis
    high_correlated_cols(df, plot=True)

    # Outlier handling
    for col in num_cols:
        if check_outlier(df, col):
            replace_with_thresholds(df, col)

    # Missing values analysis
    missing_values_table(df)
    df.dropna(inplace=True)

    # Feature engineering
    df = create_new_features(df)

    # Encoding categorical variables
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Scaling features
    num_cols = [col for col in num_cols if col != "Salary"]
    df = scale_features(df, num_cols)

    # Model training and evaluation
    y = df["Salary"]
    X = df.drop(["Salary"], axis=1)
    train_base_models(X, y)

    # Hyperparameter optimization
    regressors = [
        ("RF", RandomForestRegressor(),
         {"max_depth": [5, 8, 15, None], "max_features": [5, 7, "auto"], "min_samples_split": [8, 15, 20],
          "n_estimators": [200, 500]}),
        ('GBM', GradientBoostingRegressor(),
         {"learning_rate": [0.01, 0.1], "max_depth": [3, 8], "n_estimators": [500, 1000], "subsample": [1, 0.5, 0.7]}),
        ('LightGBM', LGBMRegressor(),
         {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500], "colsample_bytree": [0.7, 1]}),
        ("CatBoost", CatBoostRegressor(), {"iterations": [200, 500], "learning_rate": [0.01, 0.1], "depth": [3, 6]})
    ]
    best_models = hyperparameter_optimization(X, y, regressors)



    # Validation curves
    rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]], ["max_features", [3, 5, 7, "auto"]],
                     ["min_samples_split", [2, 5, 8, 15, 20]], ["n_estimators", [10, 50, 100, 200, 500]]]
    rf_model = RandomForestRegressor(random_state=17)
    for param_name, param_range in rf_val_params:
        val_curve_params(rf_model, X, y, param_name, param_range, scoring="neg_mean_absolute_error")


if __name__ == "__main__":
    main()
