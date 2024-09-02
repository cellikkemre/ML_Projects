from sklearn.linear_model import LogisticRegression
import pickle
import os


def train_logistic_regression(X, y):
    """
    Train a logistic regression model on the provided data.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.

    Returns:
    - model (LogisticRegression): Trained logistic regression model.
    """
    log_model = LogisticRegression().fit(X, y)
    return log_model


def predict_new_observation(model, X):
    """
    Predict the outcome for a new observation using the trained model.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - X (pd.DataFrame): Feature matrix for new observation(s).

    Returns:
    - array: Predicted outcome for the new observation(s).
    """
    return model.predict(X.sample(1, random_state=42))


def save_model(model, filename="logistic_regression_model.pkl"):
    """
    Save the trained model to a file.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - filename (str): The filename to save the model under.

    Returns:
    - None
    """
    # Dosya yolunu oluştur
    filepath = os.path.join("src", "models", filename)

    # Klasör mevcut değilse oluştur
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Modeli kaydet
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {filepath}")
