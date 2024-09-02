from House_Price_Prediction.src.data.load_data import load_and_combine_data, check_df
from House_Price_Prediction.src.features.feature_engineering import feature_engineering
from House_Price_Prediction.src.models.train_model import train_models, hyperparameter_optimization
from House_Price_Prediction.src.evaluation.evaluate_model import evaluate_model, save_predictions
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


def main():
    # Load and combine the train and test datasets
    df = load_and_combine_data("House_Price_Prediction/data/raw/train.csv",
                               "House_Price_Prediction/data/raw/test.csv")

    # Check the combined dataframe
    check_df(df)

    # Apply feature engineering
    df = feature_engineering(df)

    # Separate the combined dataframe into train and test sets
    train_df = df[df['SalePrice'].notnull()]
    test_df = df[df['SalePrice'].isnull()]

    y = train_df['SalePrice']
    X = train_df.drop(["SalePrice"], axis=1)

    # Train models using cross-validation and print RMSE for each model
    train_models(X, y)

    # Split the data into train and test sets for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    # Train the LightGBM model
    lgbm = LGBMRegressor().fit(X_train, y_train)

    # Hyperparameter optimization (optional)
    # Uncomment the following line to perform hyperparameter optimization:
    # lgbm = hyperparameter_optimization(X_train, y_train)

    # Evaluate the model and calculate RMSE on the test set
    rmse = evaluate_model(lgbm, X_test, y_test)
    print(f"Model RMSE: {rmse}")

    # Predict and save the results for the test set
    save_predictions(lgbm, test_df, file_name="housePricePredictions.csv")

    print("Predictions have been saved to 'housePricePredictions.csv'.")


if __name__ == "__main__":
    main()