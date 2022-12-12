import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
from catboost import CatBoostRegressor
# 导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

# 类别属性和数字属性分类
numerical_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                  'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                  'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                  'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                  'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                    'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


x = ['GrLivArea', 'GarageArea']


def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.975) - data_ser.quantile(0.025))
        val_low = data_ser.quantile(0.025) - iqr
        val_up = data_ser.quantile(0.975) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    # plt.show()
    return data_n

for a in x:
    train = outliers_proc(train, a, scale=1)
    
    
# 填充缺失值
for col in numerical_cols:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)
#     train[col].fillna(train[col].mean(), inplace=True)
#     test[col].fillna(test[col].mean(), inplace=True)
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

train = train.drop('Utilities', axis=1)
test = test.drop('Utilities', axis=1)

X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']

model = CatBoostRegressor(iterations=1000, learning_rate=0.05, loss_function='RMSE', logging_level='Silent')
model.fit(X, y, cat_features=categorical_cols, plot=True)
pred = model.predict(test)
