# Start Time : 2023/2/19 22:03
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
from sklearn.impute import KNNImputer
import warnings

from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
from datetime import datetime

# 导入数据
train = pd.read_csv("data/happiness_train_complete.csv", encoding='ISO-8859-1')
test = pd.read_csv("data/happiness_test_complete.csv", encoding='ISO-8859-1')
test_sub = pd.read_csv("data/happiness_submit.csv", encoding='ISO-8859-1')

# 查看label分布
y_train_ = train["happiness"]
# 将-8换成3
y_train_ = y_train_.map(lambda x: 3 if x == -8 else x)
# 让label从0开始
y_train_ = y_train_.map(lambda x: x - 1)
# train和test连在一起
data = pd.concat([train, test], axis=0, ignore_index=True)

# 处理时间特征
data['survey_time'] = pd.to_datetime(data['survey_time'], format='%Y-%m-%d %H:%M:%S')
data["weekday"] = data["survey_time"].dt.weekday
data["quarter"] = data["survey_time"].dt.quarter
data["hour"] = data["survey_time"].dt.hour
data["month"] = data["survey_time"].dt.month
data['weekend'] = (data.weekday >= 5).astype(int)


# 把一天的时间分段
def hour_cut(x):
    if 0 <= x < 6:
        return 0
    elif 6 <= x < 8:
        return 1
    elif 8 <= x < 12:
        return 2
    elif 12 <= x < 14:
        return 3
    elif 14 <= x < 18:
        return 4
    elif 18 <= x < 21:
        return 5
    elif 21 <= x < 24:
        return 6


data["hour_cut"] = data["hour"].map(hour_cut)

# 做问卷时候的年龄
data["survey_age"] = 2015 - data["birth"]

#

# 让label从0开始
data["happiness"] = data["happiness"].map(lambda x: x - 1)

# 去掉三个缺失值很多的
data = data.drop(["edu_other"], axis=1)
data = data.drop(["happiness"], axis=1)
data = data.drop(["survey_time"], axis=1)

# 是否入党
data["join_party"] = data["join_party"].map(lambda x: 0 if pd.isnull(x) else 1)


# 出生的年代
def birth_split(x):
    if 1920 <= x <= 1930:
        return 0
    elif 1930 < x <= 1940:
        return 1
    elif 1940 < x <= 1950:
        return 2
    elif 1950 < x <= 1960:
        return 3
    elif 1960 < x <= 1970:
        return 4
    elif 1970 < x <= 1980:
        return 5
    elif 1980 < x <= 1990:
        return 6
    elif 1990 < x <= 2000:
        return 7


data["birth_s"] = data["birth"].map(birth_split)


# 收入分组
# def income_cut(x):
#     if x < 0:
#         return 0
#     elif 0 <= x < 1200:
#         return 1
#     elif 1200 < x <= 10000:
#         return 2
#     elif 10000 < x < 24000:
#         return 3
#     elif 24000 < x < 40000:
#         return 4
#     elif 40000 <= x:
#         return 5

def income_cut(x):
    if x < 0:
        return 0
    elif x < 1200:
        return 0
    elif 1200 < x <= 10000:
        return 1
    elif 10000 < x < 24000:
        return 2
    elif 24000 < x < 40000:
        return 3
    elif 40000 <= x:
        return 4


data["income_cut"] = data["income"].map(income_cut)

# 填充数据
num_cols = list(data.select_dtypes(exclude=['object']).columns)

for n in num_cols:
    data[n] = data[n].map(lambda x: None if x == -1 else x)
    data[n] = data[n].map(lambda x: None if x == -2 else x)
    data[n] = data[n].map(lambda x: None if x == -3 else x)
    data[n] = data[n].map(lambda x: None if x == -8 else x)

# data["edu_status"] = data["edu_status"].fillna(5)
# data["edu_yr"] = data["edu_yr"].fillna(-2)
# data["property_other"] = data["property_other"].map(lambda x: 0 if pd.isnull(x) else 1)
# data["hukou_loc"] = data["hukou_loc"].fillna(1)
# data["social_neighbor"] = data["social_neighbor"].fillna(8)
# data["social_friend"] = data["social_friend"].fillna(8)
# data["work_status"] = data["work_status"].fillna(0)
# data["work_yr"] = data["work_yr"].fillna(0)
# data["work_type"] = data["work_type"].fillna(0)
# data["work_manage"] = data["work_manage"].fillna(0)
# data["family_income"] = data["family_income"].fillna(-2)
# data["invest_other"] = data["invest_other"].map(lambda x: 0 if pd.isnull(x) else 1)
data["property_other"] = data["property_other"].map(lambda x: 0 if pd.isnull(x) else 1)
data["invest_other"] = data["invest_other"].map(lambda x: 0 if pd.isnull(x) else 1)

imputer = KNNImputer(n_neighbors=6)
data[num_cols] = imputer.fit_transform(data[num_cols])

# 填充数据
# data["minor_child"] = data["minor_child"].fillna(0)
# data["marital_1st"] = data["marital_1st"].fillna(0)
# data["s_birth"] = data["s_birth"].fillna(0)
# data["marital_now"] = data["marital_now"].fillna(0)
# data["s_edu"] = data["s_edu"].fillna(0)
# data["s_political"] = data["s_political"].fillna(0)
# data["s_hukou"] = data["s_hukou"].fillna(0)
# data["s_income"] = data["s_income"].fillna(0)
# data["s_work_exper"] = data["s_work_exper"].fillna(0)
# data["s_work_status"] = data["s_work_status"].fillna(0)
# data["s_work_type"] = data["s_work_type"].fillna(0)
data['public_total'] = data['public_service_1'] + data['public_service_2'] + data['public_service_3'] + data[
    'public_service_4'] + data['public_service_5'] + data['public_service_6'] + data['public_service_9'] + data[
                           'public_service_7'] + data['public_service_8']
data['trust_total'] = data['trust_1'] + data['trust_2'] + data['trust_3'] + data['trust_4'] + data['trust_5'] + data[
    'trust_6'] + data['trust_7'] + data['trust_8'] + data['trust_9'] + data['trust_10'] + data['trust_11'] + data[
                          'trust_12'] + data['trust_13']
# print(data['public_total'][:10])
# print(data['trust_total'][:10])
# 体重（kg）÷（身高（m）x身高（m））
data['BMI'] = data['weight_jin'] / (2 * data['height_cm'] * data['height_cm']) * 10000
# print(data[['height_cm', 'weight_jin', 'BMI']][:10])

data['media_total'] = data['media_1'] + data['media_2'] + data['media_3'] + data['media_4'] + data['media_5'] + data[
    'media_6']
data['leisure_total'] = data['leisure_1'] + data['leisure_2'] + data['leisure_3'] + data['leisure_4'] + data[
    'leisure_5'] + data['leisure_6'] + data['leisure_7'] + data['leisure_8'] + data['leisure_9'] + data['leisure_10'] + \
                        data['leisure_11'] + data['leisure_12']

data['class_before_now'] = data['class'] - data['class_10_before']
data['class_now_after'] = data['class_10_after'] - data['class']
data['class_before_after'] = data['class_10_after'] - data['class_10_before']

# print(data[['class_before_now', 'class_now_after', 'class_before_after']][:10])

data['inc_distance'] = data['inc_exp'] - data['income']
data['1st_marital_time'] = data['marital_1st'] - data['birth']
data['now_marital_time'] = data['marital_now'] - data['birth']
data['f_age'] = 2015 - data['f_birth']
data['m_age'] = 2015 - data['m_birth']

data = data.drop(["id"], axis=1)

X_train_ = data[:train.shape[0]]
X_test_ = data[train.shape[0]:]

target_column = 'happiness'
feature_columns = list(X_test_.columns)

X_train = np.array(X_train_)
y_train = np.array(y_train_)
X_test = np.array(X_test_)


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

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train_)))

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
oof_lgb = np.zeros(len(X_train_))
predictions_lgb = np.zeros(len(X_test_))

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

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train_)))

from catboost import Pool, CatBoostRegressor
# cat_features=[0,2,3,10,11,13,15,16,17,18,19]
from sklearn.model_selection import train_test_split

# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train_, y_train_, test_size=0.3, random_state=2019)
# train_pool = Pool(X_train_s, y_train_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# val_pool = Pool(X_test_s, y_test_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# test_pool = Pool(X_test_ ,cat_features=[0,2,3,10,11,13,15,16,17,18,19])


kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_cb = np.zeros(len(X_train_))
predictions_cb = np.zeros(len(X_test_))
kfold = kfolder.split(X_train_, y_train_)
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
        'n_estimators': 100000,
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
    predictions_cb += model_cb.predict(X_test_, ntree_end=model_cb.best_iteration_) / kfolder.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_cb, y_train_)))

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

print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train_)))

result = list(predictions)
result = list(map(lambda x: x + 1, result))
test_sub["happiness"] = result
test_sub.to_csv("submit_20190515_2.csv", index=False)
#  0.45543947