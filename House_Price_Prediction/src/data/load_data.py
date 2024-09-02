import pandas as pd
from House_Price_Prediction.src.utils.helpers import check_df

def load_and_combine_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    df = pd.concat([train, test], ignore_index=False).reset_index(drop=True)
    df = df.drop("Id", axis=1)
    return df