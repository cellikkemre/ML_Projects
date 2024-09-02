from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_validate, train_test_split
from Diabetes_Prediction_with_Logistic_Regression.src.utils.helpers import plot_confusion_matrix
import matplotlib.pyplot as plt



def evaluate_model(model, X, y):
    """
    Evaluate the logistic regression model using test data.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.

    Returns:
    - None
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Classification report
    print(classification_report(y, y_pred))

    # Confusion matrix
    plot_confusion_matrix(y, y_pred)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    # AUC Score
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC Score: {auc_score:.4f}")


def model_validation_holdout(model, X, y):
    """
    Validate the logistic regression model using holdout method.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.

    Returns:
    - None
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Holdout Validation")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve (Holdout)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC Score (Holdout): {auc_score:.4f}")


def model_validation_cross_validation(model, X, y):
    """
    Validate the logistic regression model using 10-Fold Cross-Validation.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.

    Returns:
    - None
    """
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

    print("10-Fold Cross Validation")
    print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"Mean Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Mean Recall: {cv_results['test_recall'].mean():.4f}")
    print(f"Mean F1 Score: {cv_results['test_f1'].mean():.4f}")
    print(f"Mean ROC AUC: {cv_results['test_roc_auc'].mean():.4f}")

