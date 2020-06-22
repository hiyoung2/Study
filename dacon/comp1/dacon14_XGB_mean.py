import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape :", data.shape) # (10000, 75)
print("test.shape :", x_pred.shape)   # (10000, 71)  
print("submission.shape :", submission.shape) # (10000, 4)

# 결측치 제거

data = data.transpose()
x_pred = x_pred.transpose()

print("====================================")
print("data.shape :", data.shape) # (75, 10000)
print("x_pred.shape :", x_pred.shape) # (71, 10000)
print("====================================")

data = data.interpolate()
x_pred = x_pred.interpolate()

print("====================================")
print(data.isnull().sum()) 
print(x_pred.isnull().sum()) 
print("====================================")


data = data.fillna(data.mean())
x_pred = x_pred.fillna(x_pred.mean())

data = data.transpose()
x_pred = x_pred.transpose()

print("====================================")
print("data.shape :", data.shape) # (10000, 75)
print("x_pred.shape :", x_pred.shape) # (10000, 71)
print("====================================")



x = data.iloc[:, :-4]
y = data.iloc[:, -4:]
print("x.shape :", x.shape) # (10000, 71)
print("y.shape :", y.shape) # (10000, 4)
 

x = x.values
y = y.values
x_pred = x_pred.values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


# scaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# 2. 모델 구성

n_estimators = 300
learning_rate = 0.09
colsample_bytree = 0.9
colsample_bylevel = 0.6
max_depth = 5
n_jobs = 6

model = MultiOutputRegressor(XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, 
                     colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
                     max_depth = max_depth, n_jobs = n_jobs, cv = 5))

# model = XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, 
#                      colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
#                      max_depth = max_depth, n_jobs = n_jobs, cv = 5)


model.fit(x_train, y_train)
y_pred = model.predict(x_test)

score = model.score(x_test, y_test)

mae = mean_absolute_error(y_test, y_pred)

print("R2 :", score)
print("MAE :", mae)


'''
R2 : 0.4353875302443971
MAE : 1.20094992852658
'''


# 최종 제출 파일
submit = model.predict(x_pred)

a = np.arange(10000, 20000)
submit = pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp1/submit/XGB_mean.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label = "id")


