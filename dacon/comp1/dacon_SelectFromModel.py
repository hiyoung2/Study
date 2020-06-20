import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
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

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

print("x_train.shape :", x_train.shape) # (8000, 71)
print("x_test.shape :", x_test.shape)   # (2000, 71)
print("y_train.shape :", y_train.shape) # (8000, 4)
print("y_test.shape :", y_test.shape)   # (2000, 4)


# parameters = [
#     {'n_estimators' : [100, 200, 300], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6]},
#     {'n_estimators' : [90, 100, 110], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9]},
#     {'n_estimators' : [90, 100, 110], 'learning_rate' : [0.01, 0.09, 0.1, 0.3, 0.5],
#     'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9],
#     'colsample_bylevel' : [0.6, 0.7, 0.8, 0.9]}
# ]

model = MultiOutputRegressor(XGBRegressor(ck = 5))

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("R2 :", score)

print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_)
print(thresholds)


'''
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