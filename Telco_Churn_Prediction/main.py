
from src.data_loading import load_data
from utils.helpers import check_df, grab_col_names, cat_summary, num_summary, target_summary_with_num, \
    target_summary_with_cat
from src.data_preprocessing import *
from src.features import *
from src.modelling import *
import warnings

warnings.simplefilter(action="ignore")


def main():
    # Load data
    df = load_data("data/Telco-Customer-Churn.csv")

    # Basic data overview
    check_df(df)

    # Convert "TotalCharges" variable to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # Convert "Yes" and "No" values in the "Churn" column to 1 and 0, respectively
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # Identify categorical and numerical variables
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Analysis of categorical variables
    for col in cat_cols:
        cat_summary(df, col)

    # Analysis of numerical variables
    for col in num_cols:
        num_summary(df, col)

    # Analysis of numerical and categorical variables based on target
    for col in num_cols:
        target_summary_with_num(df, "Churn", col)
    for col in cat_cols:
        target_summary_with_cat(df, "Churn", col)

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Detect and replace outliers
    for col in num_cols:
        if check_outlier(df, col):
            replace_with_thresholds(df, col)

    # Feature engineering
    # Tenure groups
    df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
    df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
    df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
    df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
    df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
    df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

    # Other new features
    df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
    df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
                x["TechSupport"] != "Yes") else 0, axis=1)
    df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
                                           axis=1)
    df["NEW_TotalServices"] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(
        axis=1)
    df["NEW_FLAG_ANY_STREAMING"] = df.apply(
        lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
    df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
        lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)
    df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]
    df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

    # Encoding categorical variables
    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Modeling phase
    y = df["Churn"]
    X = df.drop(["Churn", "customerID"], axis=1)

    models = [('LR', LogisticRegression(random_state=12345)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=12345)),
              ('RF', RandomForestClassifier(random_state=12345)),
              ('SVM', SVC(gamma='auto', random_state=12345)),
              ('XGB', XGBClassifier(random_state=12345)),
              ("LightGBM", LGBMClassifier(random_state=12345)),
              ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

    # Feature importance
    final_model = RandomForestClassifier(random_state=17)
    final_model.fit(X, y)
    plot_importance(final_model, X)


if __name__ == "__main__":
    main()

