from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from House_Price_Prediction.src.utils.helpers import plot_importance
import numpy as np


def train_models(X, y):
    models = [('LR', LinearRegression()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor())]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


def hyperparameter_optimization(X_train, y_train):
    """
    Performs hyperparameter optimization for the LightGBM model using GridSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Target variable for training.

    Returns:
    model: LightGBM model with the best hyperparameters.
    """
    lgbm_model = LGBMRegressor(random_state=42)

    # Define the hyperparameter grid
    lgbm_params = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5, 7],
        "colsample_bytree": [0.5, 0.7, 1]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=lgbm_model,
                               param_grid=lgbm_params,
                               cv=3,
                               n_jobs=-1,
                               verbose=2,
                               scoring="neg_mean_squared_error")

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    return best_model


# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# Perform hyperparameter optimization for LightGBM
lgbm_best = hyperparameter_optimization(X_train, y_train)

# Plot feature importance
plot_importance(lgbm_best, X_train)