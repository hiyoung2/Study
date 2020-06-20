import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputEstimator
from sklearn.metrics import mean_absolute_error, r2_score

data = np.load("./data/dacon/comp1/data.npy", allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/x_pred.npy", allow_pickle = True)

print("data.shape :", data.shape)
print("x_pred.shape :", x_pred.shape)

x = data[:, :-4]
y = data[:, -4:]

print()
print("x.shape :", x.shape) # (10000, 71)
print("y.shape :", y.shape) # (10000, 4)
print()

pca = PCA(n_components=1)
pca.fit(y)
y_pca = pca.transform(y)

scaler = RobustScaler()
scaler.fit(y_pca)
y_data = scaler.transform(y_pca)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_data, test_size = 0.2, shuffle = True, random_state = 66
)

# print("x_train.shape :", x_train.shape) # (8000, 71)
# print("x_test.shape :", x_test.shape)   # (2000, 71)
# print("y_train.shape :", y_train.shape) # (8000, 4)
# print("y_test.shape :", y_test.shape)   # (2000, 4)


## PCA
print("x_train.shape :", x_train.shape) 
print("x_test.shape :", x_test.shape)   
print("y_train.shape :", y_train.shape) 
print("y_test.shape :", y_test.shape)  



# parameters = [
#     {'n_estimators' : [100, 200, 300], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6]},
#     {'n_estimators' : [90, 100, 110], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9]},
#     {'n_estimators' : [90, 100, 110], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9],
#     'colsample_bylevel' : [0.6, 0.7, 0.8, 0.9]}
# ]

# model = MultiOutputRegressor(XGBRegressor(ck = 5))
# model = RandomForestRegressor() 
# model = MultiOutputRegressor(XGBRegressor())

n_estimators = 150
learning_rate = 0.09
colsample_bytree = 0.9
colsample_bylevel = 0.6
max_depth = 8
n_jobs = -1

model = XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, 
                    colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
                    max_depth = max_depth, n_jobs = n_jobs, cv = 5)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("R2 :", score)

print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_)
print(thresholds)



for thresh in thresholds :

    selection = SelectFromModel(model, threshold = thresh, prefit = True)
    
    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh = %.3f, n = %d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))



'''
n_estimators = 150
learning_rate = 0.09
colsample_bytree = 0.9
colsample_bylevel = 0.6
max_depth = 8
n_jobs = -1

model = XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, 
                    colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
                    max_depth = max_depth, n_jobs = n_jobs, cv = 5)

not GridSearchCV

Thresh = 0.002, n = 71, R2 : 48.17%
Thresh = 0.003, n = 70, R2 : 48.64%
Thresh = 0.003, n = 69, R2 : 47.61%
Thresh = 0.004, n = 68, R2 : 48.42%
Thresh = 0.004, n = 67, R2 : 48.20%
Thresh = 0.005, n = 66, R2 : 48.59%
Thresh = 0.005, n = 65, R2 : 48.32%
Thresh = 0.005, n = 64, R2 : 47.88%
Thresh = 0.006, n = 63, R2 : 48.31%
Thresh = 0.006, n = 62, R2 : 49.13%
Thresh = 0.006, n = 61, R2 : 47.61%
Thresh = 0.006, n = 60, R2 : 48.44%
Thresh = 0.007, n = 59, R2 : 48.47%
Thresh = 0.007, n = 58, R2 : 49.02%
Thresh = 0.007, n = 57, R2 : 48.33%
Thresh = 0.007, n = 56, R2 : 48.58%
Thresh = 0.007, n = 55, R2 : 48.57%
Thresh = 0.007, n = 54, R2 : 48.48%
Thresh = 0.007, n = 53, R2 : 48.66%
Thresh = 0.007, n = 52, R2 : 48.51%
Thresh = 0.007, n = 51, R2 : 49.07%
Thresh = 0.008, n = 50, R2 : 48.22%
Thresh = 0.008, n = 49, R2 : 47.92%
Thresh = 0.008, n = 48, R2 : 48.35%
Thresh = 0.008, n = 47, R2 : 47.78%
Thresh = 0.008, n = 46, R2 : 49.08%
Thresh = 0.008, n = 45, R2 : 49.23%
Thresh = 0.008, n = 44, R2 : 47.74%
Thresh = 0.008, n = 43, R2 : 49.63%
Thresh = 0.008, n = 42, R2 : 48.74%
Thresh = 0.008, n = 41, R2 : 47.68%
Thresh = 0.009, n = 40, R2 : 48.76%
Thresh = 0.009, n = 39, R2 : 49.28%
Thresh = 0.009, n = 38, R2 : 49.11%
Thresh = 0.009, n = 37, R2 : 48.83%
Thresh = 0.009, n = 36, R2 : 49.39%
Thresh = 0.009, n = 35, R2 : 49.07%
Thresh = 0.009, n = 34, R2 : 48.89%
Thresh = 0.009, n = 33, R2 : 49.40%
Thresh = 0.010, n = 32, R2 : 50.25%
Thresh = 0.010, n = 31, R2 : 49.43%
Thresh = 0.010, n = 30, R2 : 48.49%
################################### Thresh = 0.010, n = 29, R2 : 51.05%
Thresh = 0.011, n = 28, R2 : 50.05%
Thresh = 0.012, n = 27, R2 : 49.76%
Thresh = 0.012, n = 26, R2 : 49.96%
Thresh = 0.013, n = 25, R2 : 48.93%
Thresh = 0.013, n = 24, R2 : 49.35%
Thresh = 0.013, n = 23, R2 : 48.21%
Thresh = 0.014, n = 22, R2 : 49.89%
Thresh = 0.015, n = 21, R2 : 46.02%
Thresh = 0.016, n = 20, R2 : 46.48%
Thresh = 0.016, n = 19, R2 : 45.16%
Thresh = 0.017, n = 18, R2 : 43.09%
Thresh = 0.017, n = 17, R2 : 41.87%
Thresh = 0.018, n = 16, R2 : 41.22%
Thresh = 0.020, n = 15, R2 : 40.48%
Thresh = 0.022, n = 14, R2 : 41.06%
Thresh = 0.022, n = 13, R2 : 38.34%
Thresh = 0.022, n = 12, R2 : 40.58%
Thresh = 0.023, n = 11, R2 : 38.14%
Thresh = 0.023, n = 10, R2 : 37.98%
Thresh = 0.029, n = 9, R2 : 38.01%
Thresh = 0.030, n = 8, R2 : 39.11%
Thresh = 0.031, n = 7, R2 : 35.91%
Thresh = 0.031, n = 6, R2 : 35.73%
Thresh = 0.038, n = 5, R2 : 33.05%
Thresh = 0.039, n = 4, R2 : 30.00%
Thresh = 0.046, n = 3, R2 : 26.39%
Thresh = 0.064, n = 2, R2 : 22.20%
Thresh = 0.064, n = 1, R2 : 2.90%
'''