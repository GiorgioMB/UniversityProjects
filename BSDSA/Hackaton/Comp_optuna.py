#%%
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

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

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    }

    clf = XGBClassifier(**param, n_jobs=4)
    clf.fit(X_train, y_train)
    
    pred_train_positive = clf.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, pred_train_positive)
    pred_test_positive = clf.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, pred_test_positive)

    return -roc_auc_train + abs(roc_auc_train - roc_auc_test)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(columns=['target']), dataframe['target'], test_size=0.2)

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=100)

# Output the best parameters
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))

best_params = study.best_trial.params
best_clf = XGBClassifier(**best_params, n_jobs=4)
best_clf.fit(X_train, y_train)
pred_test_positive = best_clf.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_test, pred_test_positive)
print(f'ROC AUC score for the test set: {roc_auc_test}')
# %%
