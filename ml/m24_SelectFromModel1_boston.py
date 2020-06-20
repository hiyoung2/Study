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

# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

# parameters = [
#     {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
#     "max_depth" : [4, 5, 6]},
#     {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
#     "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
#     {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
#     "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
#     "colsample_bylevel" : [0.6, 0.7, 0.9]}
#     ]
# model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.09, max_depth = 5, colsample_bylevel = 0.7, colsample_bytree = 0.9, n_jobs = -1)

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




for thresh in thresholds: # 컬럼 수만큼 돈다! 빙글 빙글
               
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    "colsample_bylevel" : [0.6, 0.7, 0.9]}
    ]

    selection_model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs= -1)
    selection_model.fit(select_x_train, y_train)
    print(model.best_params_)
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Thersh=%.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))

'''
Thersh=0.001, n = 13, R2: 93.12%
Thersh=0.004, n = 12, R2: 92.61%
Thersh=0.012, n = 11, R2: 93.46%
Thersh=0.012, n = 10, R2: 93.26%
Thersh=0.014, n = 9, R2: 90.38%
Thersh=0.015, n = 8, R2: 93.85%
Thersh=0.018, n = 7, R2: 92.60%
Thersh=0.030, n = 6, R2: 91.03%
Thersh=0.042, n = 5, R2: 91.80%
Thersh=0.052, n = 4, R2: 91.37%
Thersh=0.069, n = 3, R2: 91.75%
Thersh=0.301, n = 2, R2: 82.52%
Thersh=0.428, n = 1, R2: 70.59%
'''





# 과제
# 그리드 서치까지 엮어라
# 파라미터 : median, threshold 정리

# 데이콘 적용해라 71개 컬럼

# 월요일까지 적용한 데이콘 소스, 데이콘 성적 메일로 보내라

# 메일 제목 : 하인영 24등 
