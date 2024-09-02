from Baseball_Salary_Prediction.utils.helpers import check_df,grab_col_names,cat_summary,num_summary,target_summary_with_cat,high_correlated_cols
from Baseball_Salary_Prediction.utils.helpers import outlier_thresholds,check_outlier,replace_with_thresholds,missing_values_table
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

warnings.filterwarnings("ignore", category=UserWarning, module='pandas')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)



#############################################
#Analysis of Categorical Variables)
#############################################
for col in cat_cols:
    cat_summary(df,col,plot=True)




#############################################
# Analysis of Numerical Variables
#############################################
for col in num_cols:
    num_summary(df,col,plot=True)




#############################################
# Analysis of Target Variable
#############################################
for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


#############################################
# Analysis of Correlation
#############################################
high_correlated_cols(df,plot=True)


#############################################
# Outliers
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#############################################
# Missing Values
#############################################

missing_values_table(df)
df.dropna(inplace=True)