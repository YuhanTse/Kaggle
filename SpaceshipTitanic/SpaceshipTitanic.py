import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# 设置到太长也不好查看，按题目来
# pd.set_option('max_colwidth', 300)
# 设置1000列的时候才换行
# pd.set_option('display.width', 1000)

### data loading
train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')

### EDA
# 获取每一个group中的人数
groups_train = {}
# dataframe特定的列经过.str之后，就可以使用各种python常用的字符处理方法了
for value in train['PassengerId'].str.split('_').str[0]:
    groups_train[value] = groups_train.get(value, 0) + 1
groups_test = {}
for value in test['PassengerId'].str.split('_').str[0]:
    groups_test[value] = groups_test.get(value, 0) + 1

    
# 添加属性列：每个人所属group内的样本量
train_number = []
test_number = []
for index, value in train['PassengerId'].items():
    train_number.append(groups_train[value.split('_')[0]])
for index, value in test['PassengerId'].items():
    test_number.append(groups_test[value.split('_')[0]])
train['GroupSize'] = train_number
test['GroupSize'] = test_number


# 将cabin拆分为'Deck', 'Num','Side'
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', expand=True)
train.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)


# 分别对数值型和类别型变量中的缺失值进行填充
cat_cols = ['VIP', 'HomePlanet', 'Destination']
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
    
    
# 使用KNN对数值列进行预测
imputer = KNNImputer(n_neighbors=4)
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.fit_transform(test[num_cols])


def set_missing_value_with_rfr(data, feature="Age"):
    # 把已有的数值型特征取出来丢进RandomForgestRegressor
    features = [feature, 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df = data[features]
    known_value = df[df[feature].notnull()].values
    unknown_value = df[df[feature].isnull()].values
    y = known_value[:, 0]
    x = known_value[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    # 用得到的模型来进行未知结果预测
    predicted_results = rfr.predict(unknown_value[:, 1::])
    data.loc[(data[feature].isnull()), feature] = predicted_results
    return data

# 基于已有的数值特征，使用随机森林对缺失的年龄值进行预测
set_missing_value_with_rfr(train, "Age")
set_missing_value_with_rfr(test, "Age")


def set_missing_value_with_clf(data, target_feature="CryoSleep", features=None):
    if features is None or not (isinstance(features,list) and len(features) > 0) :
        features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    features.insert(0, target_feature)
    df = data[features]
    known_value = df[df[target_feature].notnull()].values
    unknown_value = df[df[target_feature].isnull()].values
    y = known_value[:, 0]
    x = known_value[:, 1:]
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(x, y)
    predicted_results = clf.predict(unknown_value[:, 1::])
    data.loc[(data[target_feature].isnull()), target_feature] = predicted_results
    return data

# 基于已有的数值特征，使用随机森林对缺失的休眠值进行预测
train['CryoSleep'].replace({True: 1, False: 0}, inplace=True)
test['CryoSleep'].replace({True: 1, False: 0}, inplace=True)
set_missing_value_with_clf(train, target_feature="CryoSleep")
set_missing_value_with_clf(test, target_feature="CryoSleep")
train['CryoSleep'].replace({1: True, 0: False}, inplace=True)
test['CryoSleep'].replace({1: True, 0: False}, inplace=True)


full_dataset = pd.concat([train, test])
# 年龄分桶
age_bin_num = 5
age_labels = []
for i in range(age_bin_num):
    age_labels.append('A' + str(i))
age_cut = pd.qcut(full_dataset['Age'], age_bin_num, labels=age_labels)
train = pd.concat([train, age_cut[:train.shape[0]]], axis=1)
# age_cut = pd.qcut(test['Age'], age_bin_num, labels=age_labels)
test = pd.concat([test, age_cut[train.shape[0]:]], axis=1)

# 添加属性列MaxSpends与SumSpends
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train['MaxSpends'] = train[col_to_sum].max(axis=1)
test['MaxSpends'] = test[col_to_sum].max(axis=1)
# 因为SumSpends的值很大，故对其做normalization
train['log_spend'] = np.log(train[col_to_sum].sum(axis=1) + 1)
test['log_spend'] = np.log(test[col_to_sum].sum(axis=1) + 1)

#one-hot编码
d = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}
train['Deck'].replace(d, inplace=True)
test['Deck'].replace(d, inplace=True)
train['Num'] = train['Num'].astype(float)
test['Num'] = test['Num'].astype(float)
train = pd.get_dummies(train, prefix_sep='_')
test = pd.get_dummies(test, prefix_sep='_')

### Modeling
y = train['Transported']
X = train.drop(['Transported'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42)


### training
from lightgbm import LGBMClassifier
from lightgbm import log_evaluation, early_stopping

clf = LGBMClassifier(objective='binary',
                     learning_rate=0.005,
                     n_estimators=200,
                     num_iterations=700,
                     bagging_fraction=0.8,
                     max_depth=-1,
                     )
callbacks = [log_evaluation(period=50), early_stopping(stopping_rounds=50)]
clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'val'],
        eval_metric='auc',
        callbacks=callbacks
        )


### submit
sample = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sample['Transported'] = clf.predict(test).astype(bool)
sample.to_csv('submission.csv', index=False)
