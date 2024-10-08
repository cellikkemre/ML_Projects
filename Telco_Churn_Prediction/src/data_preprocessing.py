import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
from utils.helpers import check_df,grab_col_names,cat_summary,num_summary,target_summary_with_num,target_summary_with_cat
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df.head()
df.shape
df.info()


########### Converting the Total Charges variable to a numeric variable ###########

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')


########### Convert the "Yes" and "No" values in the "Churn" column to numeric values 1 and 0, respectively. ###########
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

#################################
# Exploratory data analysis
#################################

check_df(df)

#################################
# Identifying Numerical and Categorical Variables
#################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car




#################################
# Analysis of categorical variables
#################################

for col in cat_cols:
    cat_summary(df,col)

#################################
# Analysis of numerical variables
#################################

for col in num_cols:
    num_summary(df, col, plot=False)

#################################
# Analysis of numerical variables based on target
#################################
for col in num_cols:
    target_summary_with_num(df, "Churn", col)

#################################
# Analysis of categorical variables based on target
#################################

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##################################
# Correlation
##################################

df[num_cols].corr()

df.corrwith(df["Churn"]).sort_values(ascending=False)