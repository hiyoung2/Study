from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target
# 다음과 같이 사용할 수도 있음
# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


# model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.09, max_depth = 5, colsample_bylevel = 0.7, colsample_bytree = 0.9)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R2 :", score)

# print("========================================")
# print(model.best_params_)
# print("========================================")
print(model.feature_importances_)


# feature engineering
print("========================================")
thresholds = np.sort(model.feature_importances_)
print(thresholds)
print("========================================")



import time
start = time.time()
print(start)
print()


for thresh in thresholds: # 컬럼 수만큼 돈다! 빙글 빙글
               
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    # parameters = [
    # {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    # "max_depth" : [4, 5, 6]},
    # {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    # "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    # {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    # "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    # "colsample_bylevel" : [0.6, 0.7, 0.9]}
    # ]

    # selection_model = GridSearchCV(XGBRegressor(), parameters, cv = 5)
    selection_model = XGBRegressor(n_estimators = 1000, n_jobs = 1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Thersh=%.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))


end = time.time() - start # 총 걸린 시간
print("총 걸린 시간(n_jobs = 1) : ", end)
print()



start2 = time.time()
print(start2)
print()


for thresh in thresholds: # 컬럼 수만큼 돈다! 빙글 빙글
               
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    # parameters = [
    # {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    # "max_depth" : [4, 5, 6]},
    # {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    # "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    # {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    # "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    # "colsample_bylevel" : [0.6, 0.7, 0.9]}
    # ]

    # selection_model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)
    selection_model = XGBRegressor(n_estimators = 1000, n_job = -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Thersh=%.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))


print("총 걸린 시간(n_jobs = 1) : ", end)
print()

end2 = time.time() - start # 총 걸린 시간
print("총 걸린 시간(n_jobs = -1) : ", end2)
print()