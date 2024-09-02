from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_validate, train_test_split
from Diabetes_Prediction_with_Logistic_Regression.src.utils.helpers import plot_confusion_matrix
import matplotlib.pyplot as plt
from Diabetes_Prediction_with_Logistic_Regression.src.data.load_data import load_and_preview_data
from Diabetes_Prediction_with_Logistic_Regression.src.data.preprocess_data import data_preprocessing
from Diabetes_Prediction_with_Logistic_Regression.src.models.train_model import train_logistic_regression,predict_new_observation,save_model
from Diabetes_Prediction_with_Logistic_Regression.src.evaluation.evaluate_model import evaluate_model,model_validation_holdout,model_validation_cross_validation
import os



def main():
    # Proje kök dizinine giden yolu alın
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Dosya yolunu mutlak bir yola çevirin
    filepath = os.path.join(base_dir, "data", "raw", "diabetes.csv")

    # Veri yükleme
    df = load_and_preview_data(filepath)

    # Veri ön işleme
    df = data_preprocessing(df)

    # Hedef değişkeni ve özellikleri ayırma
    y = df["Outcome"]
    X = df.drop(["Outcome"], axis=1)

    # Model eğitimi
    log_model = train_logistic_regression(X, y)

    # Modeli kaydetme
    save_model(log_model)

    # Model değerlendirme
    evaluate_model(log_model, X, y)

    # Holdout doğrulama
    model_validation_holdout(log_model, X, y)

    # 10-Fold Cross-Validation
    model_validation_cross_validation(log_model, X, y)

    # Yeni bir gözlem için tahmin
    print("Prediction for a new observation:", predict_new_observation(log_model, X))




if __name__ == "__main__":
    main()


