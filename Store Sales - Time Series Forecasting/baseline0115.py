import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

###
# DATA LOADING
###

path = '/kaggle/input/store-sales-time-series-forecasting/'
comp_dir = Path('/kaggle/input/store-sales-time-series-forecasting')
# path = 'datasets/'
# comp_dir = Path('datasets')

train = pd.read_csv(path + 'train.csv', parse_dates=['date'])
test = pd.read_csv(path + 'test.csv', parse_dates=['date'])
holidays_events = pd.read_csv(path + 'holidays_events.csv', parse_dates=['date'])
oil = pd.read_csv(path + 'oil.csv', parse_dates=['date'])
stores = pd.read_csv(path + 'stores.csv')
transactions = pd.read_csv(path + 'transactions.csv', parse_dates=['date'])

holidays_events = holidays_events[['date', 'type']]

store_sales = pd.read_csv(
    'datasets/train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date']
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
# average_sales = (
#     store_sales
#     .groupby('date').mean()
#     .squeeze()
#     .loc['2017']
# )

oil.reset_index(drop=True, inplace=True)


###
# DATA INVESTIGATION
###


###
# DATA PREPARATION
###

y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]

# Create train set & test set
fourier = CalendarFourier(freq='W', order=12)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)

# 1. Create train set
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)
X['date'] = X.index
X.reset_index(drop=True, inplace=True)

# 2. Create test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)

# Merge and mapping:

# 1. Merge holiday events
# train set
X = pd.merge(left=X, right=holidays_events, on='date', how='left')
X.rename({'type': 'is_holiday'}, axis=1, inplace=True)
X['is_holiday'] = X.is_holiday.map({'Holiday': 1, 'Additional': 1, 'Event': 1,
                                    'Bridge': 1, 'Transfer': 1}).fillna(0).astype('int8')
X.set_index('date', inplace=True)
X['day_of_week'] = X.index.dayofweek.astype('int8')
X.loc[(X['day_of_week'] == 5) | (X['day_of_week'] == 6), 'is_holiday'] = 1
X['day_of_year'] = X.index.dayofyear.astype('int16')
X.loc[X['day_of_year'] == 1, 'is_holiday'] = 0
# test set
X_test = pd.merge(left=X_test, right=holidays_events, on='date', how='left')
X_test.rename({'type': 'is_holiday'}, axis=1, inplace=True)
X_test['is_holiday'] = X_test.is_holiday.map({'Holiday': 1, 'Additional': 1, 'Event': 1,
                                              'Bridge': 1, 'Transfer': 1}).fillna(0).astype('int8')
X_test.set_index('date', inplace=True)
X_test['day_of_week'] = X_test.index.dayofweek.astype('int8')
X_test.loc[(X_test['day_of_week'] == 5) | (X_test['day_of_week'] == 6), 'is_holiday'] = 1
X_test['day_of_year'] = X_test.index.dayofyear.astype('int16')
X_test.loc[X_test['day_of_year'] == 1, 'is_holiday'] = 0

X['date'] = X.index
X['date'] = X['date'].astype(str)
X['date'] = pd.to_datetime(X['date'])
X_test['date'] = X_test.index
X_test['date'] = X_test['date'].astype(str)
X_test['date'] = pd.to_datetime(X_test['date'])

# 2. Merge oil prices
# train set
X.reset_index(drop=True, inplace=True)
X = pd.merge(left=X, right=oil, on='date', how='left')
X.rename({'dcoilwtico': 'oil_price'}, axis=1, inplace=True)
# X['oil_price'] = X['oil_price'].fillna(0)
X['oil_price'] = X['oil_price'].fillna(method='bfill')
# test set
X_test.reset_index(drop=True, inplace=True)
# X_test['oil_price'] = 0
X_test = pd.merge(left=X_test, right=oil, on='date', how='left')
X_test.rename({'dcoilwtico': 'oil_price'}, axis=1, inplace=True)
# X_test['oil_price'] = X_test['oil_price'].fillna(0)
X_test['oil_price'] = X_test['oil_price'].fillna(method='bfill')

# 3. Create on promotion feature
# train set
onpromtionTrain = train.groupby(['date']).mean().loc["2017"]['onpromotion']
onpromtionTrain = pd.DataFrame(onpromtionTrain)
onpromtionTrain['date'] = onpromtionTrain.index
onpromtionTrain.reset_index(drop=True, inplace=True)
X = pd.merge(left=X, right=onpromtionTrain, on='date', how='left')
# test set
onpromtionTest = test.groupby(['date']).mean().loc["2017"]['onpromotion']
onpromtionTest = pd.DataFrame(onpromtionTest)
onpromtionTest['date'] = onpromtionTest.index
onpromtionTest.reset_index(drop=True, inplace=True)
X_test = pd.merge(left=X_test, right=onpromtionTest, on='date', how='left')

#
X.dropna(inplace=True)
y = y.iloc[X.index]
X.drop(columns='date', inplace=True)
X_test.drop(columns=['date'], inplace=True)

# print(X)
# print(X_test)
# y.to_csv("results/y.csv", index=False)
# X.to_csv("results/x.csv", index=False)
# X_test.to_csv("results/x_test.csv", index=False)


###
# MODELING
###

params = {
    'learning_rate': [0.001, 0.1, 0.3, 0.5, 1],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 7]
}
model = XGBRegressor(n_estimators=10, random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(model, param_distributions=params, scoring='neg_mean_absolute_error', cv=3,
                                   random_state=42, n_jobs=-1)
random_search.fit(X, y)
y_test = random_search.predict(X_test)
y_test = y_test.reshape(28512, 1)

###
# SUBMIT
###

submission = pd.DataFrame()
submission['id'] = test['id']
submission['sales'] = y_test
submission.to_csv("results/submission.csv", index=False)
