# Start Time : 2023/2/20 15:58
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# 设置到太长也不好查看，按题目来
pd.set_option('max_colwidth', 300)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

job = {'unknown': None, 'admin.': 1, 'blue-collar': 2, 'technician': 3, 'services': 4, 'management': 5,
       'retired': 6, 'entrepreneur': 7, 'self-employed': 8, 'housemaid': 9, 'unemployed': 10, 'student': 0}

train_csv['job'].replace(job, inplace=True)
test_csv['job'].replace(job, inplace=True)

marital = {'unknown': None, 'married': 0, 'single': 1, 'divorced': 2}
train_csv['marital'].replace(marital, inplace=True)
test_csv['marital'].replace(marital, inplace=True)

education = {'unknown': None, 'university.degree': 0, 'high.school': 1, 'basic.9y': 2,
             'professional.course': 3, 'basic.4y': 4, 'basic.6y': 5, 'illiterate': 6}

train_csv['education'].replace(education, inplace=True)
test_csv['education'].replace(education, inplace=True)

d = {'unknown': None, 'no': 0, 'yes': 1}
train_csv['default'].replace(d, inplace=True)
test_csv['default'].replace(d, inplace=True)

train_csv['housing'].replace(d, inplace=True)
test_csv['housing'].replace(d, inplace=True)

train_csv['loan'].replace(d, inplace=True)
test_csv['loan'].replace(d, inplace=True)

poutcome = {'nonexistent': 2, 'failure': 0, 'success': 1}
train_csv['poutcome'].replace(poutcome, inplace=True)
test_csv['poutcome'].replace(poutcome, inplace=True)

day_of_week = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
train_csv['day_of_week'].replace(day_of_week, inplace=True)
test_csv['day_of_week'].replace(day_of_week, inplace=True)

month = {'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6,
         'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
train_csv['month'].replace(month, inplace=True)
test_csv['month'].replace(month, inplace=True)

contact = {'cellular': 0, 'telephone': 1}
train_csv['contact'].replace(contact, inplace=True)
test_csv['contact'].replace(contact, inplace=True)

full_dataset = pd.concat([train_csv, test_csv], axis=0)
# 年龄分桶
age_bin_num = 5
age_labels = []
for i in range(age_bin_num):
    age_labels.append(i)
age_cut = pd.qcut(full_dataset['age'], age_bin_num, labels=age_labels)
age_cut = pd.DataFrame(age_cut)
age_cut.rename(columns={'age': 'age_bin'}, inplace=True)
train_csv = pd.concat([train_csv, age_cut[:train_csv.shape[0]]], axis=1)
test_csv = pd.concat([test_csv, age_cut[train_csv.shape[0]:]], axis=1)

train_csv['duration'] = train_csv['duration']/60
test_csv['duration'] = test_csv['duration']/60


# train_csv = pd.get_dummies(train_csv, columns=["age_bin", 'marital'])
# test_csv = pd.get_dummies(test_csv, columns=["age_bin", 'marital'])
# print(train_csv[:10])
id, label = 'id', 'subscribe'
#
train_data = TabularDataset(train_csv)
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))
preds = predictor.predict(test_csv.drop(columns=[id]))
submission = pd.DataFrame({id: test_csv[id], label: preds})
submission.to_csv('submission.csv', index=False)
