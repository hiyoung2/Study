from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score


iris = load_iris()
x = iris.data
y = iris.target

# x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


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

    parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.8, 0.9]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
    "colsample_bylevel" : [0.6, 0.7, 0.8, 0.9]}
    ]

    selection_model = GridSearchCV(XGBClassifier(), parameters, cv = 5, n_jobs= -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Thersh=%.3f, n = %d, ACC: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))


'''
Thersh=0.044, n = 4, ACC: 94.97%
Thersh=0.138, n = 3, ACC: 94.97%
Thersh=0.360, n = 2, ACC: 94.97%
Thersh=0.458, n = 1, ACC: 89.93%
'''