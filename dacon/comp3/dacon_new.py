import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

x_data = np.load("./data/dacon/comp3/x_data.npy", allow_pickle = True)
y_data = np.load("./data/dacon/comp3/y_data.npy", allow_pickle = True)
x_pred = np.load("./data/dacon/comp3/x_pred.npy", allow_pickle = True)


print("x_data.shape :", x_data.shape)    # (1050000, 5)
print("y_data.shape :", y_data.shape)    # (2800, 4)
print("x_pred.shape :", x_pred.shape)    # (262500, 5)


x_data = x_data[:, 1:] # index_col = 0으로 설정헀으므로 [0]번째 컬럼은 time이 되므로 [1]번째 컬럼부터 x_data로 사용
x_pred = x_pred[:, 1:]

print("데이터 슬라이싱")
print("x_data.shape :", x_data.shape) # (1050000, 4)
print("x_pred.shape :", x_pred.shape) # (262500, 4)

x_data = x_data.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, KFold
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2, shuffle = True, random_state = 66
)

# n_estimators = 150
# learning_rate = 0.06
# colsample_bytree = 0.9
# colsample_bylevel = 0.6
# max_depth = 9
# n_jobs = -1


# parameters = {
#     "n_estimators" : [10, 100, 20, 200, 30, 300], "learning_rate" : [0.1, 0.01, 0.09], 
#     "colsample_bytree" : [0.6, 0.7, 0.8, 0.9], "colsample_bylevel" : [0.6, 0.7, 0.8, 0.9],
#     "max_depth" : [3, 4, 5, 6], "n_jobs" : [-1]}

# parameters = {
#     "n_estimators" : [10], "learning_rate" : [0.09], 
#     "colsample_bytree" : [0.6], "colsample_bylevel" : [0.6],
#     "max_depth" : [4], "n_jobs" : [-1]}


# model = MultiOutputRegressor(XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate,
#                              colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
#                              max_depth = max_depth, n_jobs = n_jobs, cv = 5))

kfold = KFold(n_splits = 5, shuffle = True) 
model = MultiOutputRegressor(XGBRegressor())


model.fit(x_train, y_train)

score = model.score(x_test, y_test)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 :", score)
print("MSE :", mse)
print("MAE :", mae)

'''
parameters = {
    "n_estimators" : [10, 100, 20, 200, 30, 300], "learning_rate" : [0.1, 0.01, 0.09], 
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9], "colsample_bylevel" : [0.6, 0.7, 0.8, 0.9],
    "max_depth" : [3, 4, 5, 6], "n_jobs" : [-1]}

search = RandomizedSearchCV(model(), parameters, cv = kfold)

search.fit(x_train, y_train)

score = search.score(x_test, y_test)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)



print("R2 :", score)
print("MSE :", mse)
print("MAE :", mae)
'''
'''
submit = model.predict(x_pred)

a = np.arange(2800, 3500)
submit = pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp3/submit_new.csv", header = ["X", "Y", "M", "V"], index = True, index_label="id")
'''