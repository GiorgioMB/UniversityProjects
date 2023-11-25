#%%
from pyparsing import col
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

dataframe = pd.read_csv('train.csv')
test_df = pd.read_csv('test_no_tgt.csv')
dataframe = dataframe.drop(columns=['cat_1','cont_9','cont_10'])
dataframe['cont_5'] = dataframe['cont_5'].fillna(dataframe['cont_5'].median())
dataframe['cont_8'] = dataframe['cont_8'].fillna(dataframe['cont_8'].median())
dataframe['cont_11'] = dataframe['cont_11'].fillna(dataframe['cont_11'].median())
dataframe['cont_13'] = dataframe['cont_13'].fillna(dataframe['cont_13'].median())
mask = dataframe['cat_2'] == 'S1'
dataframe.loc[mask,'cat_2'] = 0
mask2 = dataframe['cat_2'] == 'S2'
dataframe.loc[mask2,'cat_2'] = 1
test_df = test_df.drop(columns=['cat_1','cont_9','cont_10'])
test_df['cont_5'] = test_df['cont_5'].fillna(test_df['cont_5'].median())
test_df['cont_8'] = test_df['cont_8'].fillna(test_df['cont_8'].median())
test_df['cont_11'] = test_df['cont_11'].fillna(test_df['cont_11'].median())
test_df['cont_13'] = test_df['cont_13'].fillna(test_df['cont_13'].median())
mask = test_df['cat_2'] == 'S1'
test_df.loc[mask,'cat_2'] = 0
mask2 = test_df['cat_2'] == 'S2'
test_df.loc[mask2,'cat_2'] = 1
dataframe = dataframe.drop(columns=['ID'])
test_df = test_df.drop(columns=['ID'])
#print(dataframe['cat_2'].head(10))
#print(dataframe.isna().sum())
categorical_features = dataframe.select_dtypes(include=['object']).columns

for column in categorical_features:
    le = LabelEncoder()
    dataframe[column] = le.fit_transform(dataframe[column])
    if column in test_df.columns:
        test_df[column] = le.transform(test_df[column])
#print(sum(dataframe['target']/15296))
extimator = XGBClassifier(n_estimators=313, learning_rate=0.04739413249200394, subsample = 0.8671713611539302, 
                          colsample_bytree = 0.8685007893738412,  n_jobs=4, max_depth = 3,
                          min_child_weight = 4, colsample_bylevel = 0.932968077860579)
X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(columns =['target']), dataframe['target'], test_size=.2)
extimator.fit(X_train, y_train)
pred_test_positive = extimator.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_test, pred_test_positive)

pred_train_positive = extimator.predict_proba(X_train)[:, 1]
roc_auc_train = roc_auc_score(y_train, pred_tre_positive)



dataframe_final = pd.DataFrame(pred_test, columns=['0','1'])
test_df = pd.read_csv('test_no_tgt.csv')
test_df['predicted_proba'] = dataframe_final['1']
test_df.to_csv('submission.csv', index=False)

# %%
