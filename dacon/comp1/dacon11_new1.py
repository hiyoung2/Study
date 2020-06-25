import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor, plot_importance
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

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

# print("x_train.shape :", x_train.shape) # (8000, 71)
# print("x_test.shape :", x_test.shape)   # (2000, 71)
# print("y_train.shape :", y_train.shape) # (8000, 4)
# print("y_test.shape :", y_test.shape)   # (2000, 4)

# 2. 모델 구성

n_estimators = 5000
learning_rate = 0.01
colsample_bytree = 0.6
colsample_bylevel = 0.6
max_depth = 8
n_jobs = -1

model = MultiOutputRegressor(XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, 
                                          colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
                                          max_depth = max_depth, n_jobs = n_jobs, cv = 5, gpu_id = 0, tree_method = 'gpu_hist'))

# XGBRegressor(gpu_id=0, tree_method='gpu_hist')


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)

print("R2 :", score)
print("mae :", mae)

'''
R2 : 0.31371972630857015
mae : 1.3680957676981387
'''

# print(model.feature_importances_)
# plot_importance(model)
# plt.show()

# 최종 제출 파일
submit = model.predict(x_pred)

a = np.arange(10000, 20000)
submit = pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp1/submit/0624/submit_new1.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label = "id")


