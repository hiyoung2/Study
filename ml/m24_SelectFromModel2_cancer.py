from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score


cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    "colsample_bylevel" : [0.6, 0.7, 0.9]}
    ]
# model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.09, max_depth = 5, colsample_bylevel = 0.7, colsample_bytree = 0.9, n_jobs = -1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("ACC :", score)

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

    selection_model = GridSearchCV(XGBClassifier(), parameters, cv = 5, n_jobs= -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Thersh=%.3f, n = %d, ACC: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))



'''
Thersh=0.000, n = 30, ACC: 84.76%
Thersh=0.001, n = 29, ACC: 88.57%
Thersh=0.004, n = 28, ACC: 88.57%
Thersh=0.005, n = 27, ACC: 84.76%
Thersh=0.005, n = 26, ACC: 88.57%
Thersh=0.005, n = 25, ACC: 88.57%
Thersh=0.005, n = 24, ACC: 73.34%
Thersh=0.006, n = 23, ACC: 88.57%
Thersh=0.006, n = 22, ACC: 84.76%
Thersh=0.008, n = 21, ACC: 88.57%
Thersh=0.009, n = 20, ACC: 88.57%
Thersh=0.010, n = 19, ACC: 96.19%
Thersh=0.011, n = 18, ACC: 84.76%
Thersh=0.011, n = 17, ACC: 84.76%
Thersh=0.014, n = 16, ACC: 80.96%
Thersh=0.014, n = 15, ACC: 88.57%
Thersh=0.014, n = 14, ACC: 96.19%
Thersh=0.014, n = 13, ACC: 92.38%
Thersh=0.016, n = 12, ACC: 88.57%
Thersh=0.017, n = 11, ACC: 84.76%
Thersh=0.018, n = 10, ACC: 88.57%
Thersh=0.021, n = 9, ACC: 88.57%
Thersh=0.023, n = 8, ACC: 84.76%
Thersh=0.029, n = 7, ACC: 88.57%
Thersh=0.038, n = 6, ACC: 77.15%
Thersh=0.055, n = 5, ACC: 84.76%
Thersh=0.069, n = 4, ACC: 77.15%
Thersh=0.109, n = 3, ACC: 58.10%
Thersh=0.174, n = 2, ACC: 54.29%
Thersh=0.289, n = 1, ACC: 58.10%
'''