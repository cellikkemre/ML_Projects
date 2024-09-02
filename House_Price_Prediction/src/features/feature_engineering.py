import numpy as np
from House_Price_Prediction.src.utils.helpers import grab_col_names, outlier_thresholds, check_outlier, replace_with_thresholds, \
    missing_values_table, quick_missing_imp, rare_analyser, rare_encoder


def feature_engineering(df):
    # Sütun isimlerini alma
    cat_cols, cat_but_car, num_cols = grab_col_names(df)

    # Aykırı değerleri kontrol etme ve baskılama
    for col in num_cols:
        if col != "SalePrice":
            if check_outlier(df, col):
                replace_with_thresholds(df, col)

    # Eksik değerlerin doldurulması
    missing_values_table(df)
    df = quick_missing_imp(df, num_method="median", cat_length=17)

    # Rare analiz ve encoder
    rare_analyser(df, "SalePrice", cat_cols)
    df = rare_encoder(df, 0.01)

    return df