from sklearn.metrics import mean_squared_error
from House_Price_Prediction.src.utils.helpers import plot_importance

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    new_y = np.expm1(y_pred)
    new_y_test = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(new_y_test, new_y))
    return rmse

def save_predictions(model, test_df, file_name="housePricePredictions.csv"):
    predictions = model.predict(test_df.drop(["Id", "SalePrice"], axis=1))
    dictionary = {"Id": test_df.index, "SalePrice": predictions}
    dfSubmission = pd.DataFrame(dictionary)
    dfSubmission.to_csv(file_name, index=False)

# Model değerlendirme ve sonuçları kaydetme
rmse = evaluate_model(lgbm, X_test, y_test)
save_predictions(lgbm, test_df)