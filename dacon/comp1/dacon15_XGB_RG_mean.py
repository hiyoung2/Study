import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel

# 1. 데이터 준비
data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape :", data.shape) # (10000, 75)
print("test.shape :", x_pred.shape)   # (10000, 71)  
print("submission.shape :", submission.shape) # (10000, 4)

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

print()

# 판다스 데이터 프레임 형태에서 슬라이싱하기(iloc 이용)
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
model = MultiOutputRegressor(XGBRegressor(n_jobs = 6))

model.fit(x_train, y_train)

print(model.estimators_)
print(len(model.estimators_))



for i in range(len(model.estimators_)) :
    threshold = np.sort(model.estimators_[i].feature_importances_)

    for thresh in threshold :

        selection = SelectFromModel(model.estimators_[i], threshold = thresh, prefit = True)
       
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)

        parameters = {
            "n_estimators" : [100, 200, 300, 400, 500],
            "learning_rate" : [0.01, 0.03, 0.05, 0.07, 0.09],
            "colsample_bytree" : [0.6, 0.7, 0.8, 0.9],
            "colsample_bylevel" : [0.6, 0.7, 0.8, 0.9],
            "max_depth" : [3, 4, 5, 6]
        } 

        search = RandomizedSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)

        m_search = MultiOutputRegressor(search, n_jobs = -1)
        m_search.fit(select_x_train, y_train)

        y_pred = m_search.predict(select_x_test)

        score = m_search.score(select_x_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)

        select_x_pred = selection.transform(x_pred)

        submit = m_search.predict(select_x_pred)

        print("Threshold = %.3f, n = %d, R2 : %.2f%%, MAE : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0, mae))

        a = np.arange(10000, 20000)
        submit = pd.DataFrame(submit, a)
        submit.to_csv("./dacon/comp1/submit/submit_XGB_RG_mean_%i_%.4f.csv"%(i, mae), header = ["hhb", "hbo2", "ca", "na"], index = True, index_label = "id")

