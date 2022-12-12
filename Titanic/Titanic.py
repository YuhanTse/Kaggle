import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
import warnings

warnings.simplefilter(action='ignore')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# 设置到太长也不好查看，按题目来
pd.set_option('max_colwidth', 1000)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full_data = pd.concat([train, test], axis=0, ignore_index=True)
# 改性别
full_data.Sex = full_data.Sex.map({'male': 0, 'female': 1}).astype('int')

full_data['Family_Size'] = full_data.SibSp + full_data.Parch + 1

full_data['Title'] = full_data['Name']

for name_string in full_data['Name']:
    full_data['Title'] = full_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
           'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

full_data.replace({'Title': mapping}, inplace=True)
full_data['Last_Name'] = full_data['Name'].apply(lambda x: str.split(x, ",")[0])
full_data['Fare'].fillna(full_data['Fare'].mean(), inplace=True)
full_data['Family_Survival'] = 0.5

for grp, grp_df in full_data[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                              'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if len(grp_df) != 1:
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if smax == 1.0:
                full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif smin == 0.0:
                full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in full_data.groupby('Ticket'):
    if len(grp_df) != 1:
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if smax == 1.0:
                    full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif smin == 0.0:
                    full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = 0

full_data['Fare'].fillna(full_data['Fare'].median(), inplace=True)

full_data['FareBin'] = pd.qcut(full_data['Fare'], 5)

# 转换为0-n
label = LabelEncoder()
full_data['FareBin_Code'] = label.fit_transform(full_data['FareBin'])

full_data = full_data.drop(['Fare'], axis=1)

# filling missing values in 'age' column
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:
    age_to_impute = full_data.groupby('Title')['Age'].median()[titles.index(title)]
    full_data.loc[(full_data['Age'].isnull()) & (full_data['Title'] == title), 'Age'] = age_to_impute

full_data['AgeBin'] = pd.qcut(full_data['Age'], 4)
# 转换为0-n
label = LabelEncoder()
full_data['AgeBin_Code'] = label.fit_transform(full_data['AgeBin'])
# 0 - 4
#
full_data['Age_Sex'] = 0
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 0) & (full_data['Sex'] == 0)] = 0
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 0) & (full_data['Sex'] == 1)] = 0
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 1) & (full_data['Sex'] == 0)] = 1
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 1) & (full_data['Sex'] == 1)] = 2
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 2) & (full_data['Sex'] == 0)] = 1
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 2) & (full_data['Sex'] == 1)] = 2
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 3) & (full_data['Sex'] == 0)] = 0
full_data['Age_Sex'][(full_data['AgeBin_Code'] == 3) & (full_data['Sex'] == 1)] = 0

full_data = full_data.drop(['Name', 'Last_Name', 'Ticket', 'Title', 'FareBin', 'AgeBin', 'Cabin'], axis=1)
print(full_data[:10])
full_data.Embarked.fillna('S', inplace=True)
mapping = {'S': 0, 'C': 1, 'Q': 2}
full_data.replace({'Embarked': mapping}, inplace=True)

X_train = full_data[:len(train)].drop('Survived', axis=1)
X_test = full_data[len(train):].drop('Survived', axis=1)
y_train = train['Survived']

kfold = StratifiedKFold(n_splits=8)
XGB = XGBClassifier()

xgb_param_grid = {'learning_rate': [0.05, 0.1],
                  'reg_lambda': [0.3, 0.5],
                  'gamma': [0.8, 1],
                  'subsample': [0.8, 1],
                  'max_depth': [2, 3],
                  'n_estimators': [200, 300]
                  }

xgb_param_grid_best = {'learning_rate': [0.1],
                       'reg_lambda': [0.3],
                       'gamma': [1],
                       'subsample': [0.8],
                       'max_depth': [2],
                       'n_estimators': [300]
                       }

gs_xgb = GridSearchCV(XGB, param_grid=xgb_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs=4, verbose=1)

gs_xgb.fit(X_train, y_train)
XGB.fit(X_train, y_train)

xgb_best = gs_xgb.best_estimator_
print(xgb_best)


def CVScore(classifiers):
    cv_score = []
    names = []

    for n_classifier in range(len(classifiers)):
        name = classifiers[n_classifier][0]
        model = classifiers[n_classifier][1]
        cv_score.append(cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=kfold, n_jobs=4))
        names.append(name)

    cv_means = []

    for cv_result in cv_score:
        cv_means.append(cv_result.mean())

    cv_res = pd.DataFrame({"Model": names, "CVMeans": cv_means})
    cv_res = cv_res.sort_values("CVMeans", axis=0, ascending=False, inplace=False).reset_index(drop=True)
    print('\n-------------------------CrossVal Training scores-------------------------\n\n', cv_res)


# clf_list = [("BestRandomForest", rf_best), ("BestGradientBoost", gb_best), ("BestKNN", knn_best), ("BestXGB", xgb_best),
#            ("RandomForest", RFC), ("GradientBoost", GB), ("KNN Model 1", KNN), ("XGB", XGB), ("Best Model: KNN", knn1)]
clf_list = [("BestXGB", xgb_best)]

CVScore(clf_list)

results = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': XGB.predict(X_test)})
# results.to_csv("Titanic_prediction.csv", index=False)
actual_result = pd.read_csv('Real.csv')
from sklearn.metrics import mean_absolute_error

result = mean_absolute_error(actual_result['Survived'], results['Survived'])
print(1 - result)

