# Start Time : 2023/2/21 18:31
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# 设置到太长也不好查看，按题目来
pd.set_option('max_colwidth', 300)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

full_dataset = pd.concat([train, test], axis=0)

# print(full_dataset.isnull().sum())
# fraud                          300
# print(full_dataset['property_damage'].value_counts())
p = {'?': None, 'NO': 0, 'YES': 1}
full_dataset['police_report_available'].replace(p, inplace=True)
full_dataset['property_damage'].replace(p, inplace=True)
collision_type = {'?': None, 'Rear Collision': 0, 'Side Collision': 1, 'Front Collision': 2}
full_dataset['collision_type'].replace(collision_type, inplace=True)


# print(full_dataset['collision_type'].value_counts())


# collision_type
# 字典编码函数
def change_object_cols(se):
    value = se.unique().tolist()
    # value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


col = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies',
       'insured_relationship', 'authorities_contacted', 'incident_state', 'incident_city',
       'auto_make', 'auto_model', 'police_report_available', 'incident_type',
       'incident_severity', 'policy_csl']

for c in col:
    # print(full_dataset[c].value_counts())
    full_dataset[c] = change_object_cols(full_dataset[c])

# print('--------------')
# for c in col:
#     print(full_dataset[c].value_counts())

full_dataset['policy_bind_date'] = pd.to_datetime(full_dataset['policy_bind_date'])
full_dataset['pbd_month'] = full_dataset['policy_bind_date'].dt.month
full_dataset['pbd_year'] = full_dataset['policy_bind_date'].dt.year
full_dataset = full_dataset.drop('policy_bind_date', axis=1)
full_dataset['auto_pbd_year'] = full_dataset['pbd_year'] - full_dataset['auto_year']

col2 = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies',
        'insured_relationship', 'authorities_contacted', 'incident_state', 'incident_city',
        'auto_make', 'auto_model', 'police_report_available', 'incident_type',
        'incident_severity', 'policy_csl', 'police_report_available', 'property_damage', 'collision_type']

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=4)
full_dataset[col2] = imputer.fit_transform(full_dataset[col2])
# print(full_dataset.isnull().sum())
# 年龄分桶
age_bin_num = 5
age_labels = []
for i in range(age_bin_num):
    age_labels.append(i)
age_cut = pd.qcut(full_dataset['age'], age_bin_num, labels=age_labels)
age_cut = pd.DataFrame(age_cut)
age_cut.rename(columns={'age': 'age_bin'}, inplace=True)
full_dataset = pd.concat([full_dataset, age_cut], axis=1)
# print(full_dataset[:10])
print(full_dataset)

full_dataset = full_dataset.drop('policy_id', axis=1)
full_dataset = full_dataset.drop('incident_date', axis=1)
train = full_dataset[:train.shape[0]]
test = full_dataset[train.shape[0]:]

test = test.drop('fraud', axis=1)
target_column = 'fraud'
feature_columns = list(test.columns)
y_train = train["fraud"]
train = train.drop('fraud', axis=1)

X_train = np.array(train)
y_train = np.array(y_train)
X_test = np.array(test)

from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, RepeatedKFold


# 自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score


##### xgb

xgb_params = {"booster": 'gbtree', 'eta': 0.005, 'max_depth': 5, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

##### lgb

param = {'boosting_type': 'gbdt',
         'num_leaves': 20,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 6,
         'learning_rate': 0.01,
         "min_child_samples": 30,

         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    # print(trn_idx)
    # print(".............x_train.........")
    # print(X_train[trn_idx])
    #  print(".............y_train.........")
    #  print(y_train[trn_idx])
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])

    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))

from catboost import Pool, CatBoostRegressor
# cat_features=[0,2,3,10,11,13,15,16,17,18,19]
from sklearn.model_selection import train_test_split

# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train_, y_train_, test_size=0.3, random_state=2019)
# train_pool = Pool(X_train_s, y_train_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# val_pool = Pool(X_test_s, y_test_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# test_pool = Pool(X_test_ ,cat_features=[0,2,3,10,11,13,15,16,17,18,19])


kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_cb = np.zeros(len(X_train))
predictions_cb = np.zeros(len(X_test))
kfold = kfolder.split(X_train, y_train)
fold_ = 0
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train, y_train, test_size=0.3, random_state=2019)
for train_index, vali_index in kfold:
    print("fold n°{}".format(fold_))
    fold_ = fold_ + 1
    k_x_train = X_train[train_index]
    k_y_train = y_train[train_index]
    k_x_vali = X_train[vali_index]
    k_y_vali = y_train[vali_index]
    cb_params = {
        'n_estimators': 10000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': 0.05,
        'depth': 5,
        'use_best_model': True,
        'subsample': 0.6,
        'bootstrap_type': 'Bernoulli',
        'reg_lambda': 3
    }
    model_cb = CatBoostRegressor(**cb_params)
    # train the model
    model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=100, early_stopping_rounds=50)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(X_test, ntree_end=model_cb.best_iteration_) / kfolder.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_cb, y_train)))

from sklearn import linear_model

# 将lgb和xgb和ctb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_cb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = linear_model.BayesianRidge()
    # clf_3 =linear_model.Ridge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train)))

result = list(predictions)
result = list(map(lambda x: x + 1, result))
test_sub = pd.read_csv("data/submission.csv")
test_sub["fraud"] = result
test_sub.to_csv("submit_20190515_2.csv", index=False)