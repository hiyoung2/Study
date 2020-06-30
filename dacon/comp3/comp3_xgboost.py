import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse


x = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
y = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp3/test_features.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp3/sample_submission.csv", header = 0, index_col = 0)


print("x.shape :", x.shape)    # (1050000, 5)
print("y.shape :", y.shape)    # (2800, 4)
print("x_pred.shape :", x_pred.shape)    # (262500, 5)
print("submission :", submission.shape)  # (700, 4)


print(x.isnull().sum())
'''
Time    0
S1      0
S2      0
S3      0
S4      0
'''

print(x.info())
print(x.describe())
print("===========================================================================================")

# time data를 일단 슬라이싱
x = x.values[:, 1:]
y = y.values
x_pred = x_pred.values[:, 1:]


print("x.shape :", x.shape) # x_shape : (1050000, 4)
print("x_pred.shape :", x_pred.shape) # x_pred.shape : (262500, 4)
print("y.shape :", y.shape) # (2800, 4)
print("submission.shape :", submission.shape) # (700, 4)
print("===========================================================================================")


x = x.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)



x_train, x_test, y_train , y_test = tts(x, y, test_size = 0.2, random_state = 66, shuffle = True)


print("x_train.shape :", x_train.shape) # (2240, 1500)
print("x_test.shape :", x_test.shape)   # (560, 1500)
print("y_train.shape :", y_train.shape) # (2240, 4)
print("y_test.shape :", y_test.shape)   # (560, 4)


xgb_0 = XGBRegressor()
xgb_1 = XGBRegressor()
xgb_2 = XGBRegressor()
xgb_3 = XGBRegressor()

y_train_0 = y_train[:, 0]
y_train_1 = y_train[:, 1]
y_train_2 = y_train[:, 2]
y_train_3 = y_train[:, 3]

y_test_0 = y_test[:, 0]
y_test_1 = y_test[:, 1]
y_test_2 = y_test[:, 2]
y_test_3 = y_test[:, 3]

# LGBM Model Fit

xgb_0.fit(x_train, y_train_0)
xgb_1.fit(x_train, y_train_1)
xgb_2.fit(x_train, y_train_2)
xgb_3.fit(x_train, y_train_3)

score_0 = xgb_0.score(x_test, y_test_0)
score_1 = xgb_1.score(x_test, y_test_1)
score_2 = xgb_2.score(x_test, y_test_2)
score_3 = xgb_3.score(x_test, y_test_3)

print("r2_0 : ", score_0)
print("r2_1 : ", score_1)
print("r2_2 : ", score_2)
print("r2_3 : ", score_3)


# print("lgbm_0's feature importances")
# print(lgbm_0.feature_importances_)

# print("lgbm_1's feature importances")
# print(lgbm_1.feature_importances_)

# print("lgbm_2's feature importances")
# print(lgbm_2.feature_importances_)

# print("lgbm_3's feature importances")
# print(lgbm_3.feature_importances_)


# plot_importance(lgbm_0)
# plot_importance(lgbm_1)
# plot_importance(lgbm_2)
# plot_importance(lgbm_3)

# plt.show()

thresholds_0 = np.sort(xgb_0.feature_importances_)
thresholds_1 = np.sort(xgb_1.feature_importances_)
thresholds_2 = np.sort(xgb_2.feature_importances_)
thresholds_3 = np.sort(xgb_3.feature_importances_)

# print(thresholds_0)
# print()

# import statistics
# median_0 = statistics.median(thresholds_0)
# print(median_0)
# print()

# median_3 = statistics.median(thresholds_0)
# print(median_3)
# print("thresholds 평균값")
# print(sum(thresholds_0, 0.0)/len(thresholds_0)) # 2.078666666666667
# print(sum(thresholds_1, 0.0)/len(thresholds_1)) # 1.946
# print(sum(thresholds_2, 0.0)/len(thresholds_2)) # 2.1773333333333333
# print(sum(thresholds_3, 0.0)/len(thresholds_3)) # 2.0713333333333335


select_0 = SelectFromModel(xgb_0, threshold=thresholds_0[1], prefit = True)
select_1 = SelectFromModel(xgb_1, threshold=thresholds_1[2], prefit = True)
select_2 = SelectFromModel(xgb_2, threshold=thresholds_2[5], prefit = True)
select_3 = SelectFromModel(xgb_3, threshold=thresholds_3[10], prefit = True)

select_xgb_0 = XGBRegressor(n_estimators = 100, learning_rate = 0.03, max_depth = 4)
select_xgb_1 = XGBRegressor(n_estimators = 200, learning_rate = 0.05, max_depth = 5)
select_xgb_2 = XGBRegressor(n_estimators = 300, learning_rate = 0.07, max_depth = 6)
select_xgb_3 = XGBRegressor(n_estimators = 400, learning_rate = 0.09, max_depth = 7)

parameters = {}

# parameter = [
#             {"n_estimators":[100, 200, 300], "learning_rate" : [0.01, 0.03, 0.05, 0.07, 0.09],
#             "max_depth" : [6, 7, 8], "colsample_bytree":[0.6, 0.7, 0.8, 0.9], "colsample_bylevel":[0.6, 0.7, 0.8, 0.9]}
#             ]


search_xgb_0 = GridSearchCV(select_xgb_0, {}, cv = 5, n_jobs = -1)
search_xgb_1 = GridSearchCV(select_xgb_1, {}, cv = 5, n_jobs = -1)
search_xgb_2 = GridSearchCV(select_xgb_2, {}, cv = 5, n_jobs = -1)
search_xgb_3 = GridSearchCV(select_xgb_3, {}, cv = 5, n_jobs = -1)


select_x_train_0 = select_0.transform(x_train)
select_x_train_1 = select_1.transform(x_train)
select_x_train_2 = select_2.transform(x_train)
select_x_train_3 = select_3.transform(x_train)


select_x_test_0 = select_0.transform(x_test)
select_x_test_1 = select_1.transform(x_test)
select_x_test_2 = select_2.transform(x_test)
select_x_test_3 = select_3.transform(x_test)

x_pred_0 = select_0.transform(x_pred)
x_pred_1 = select_1.transform(x_pred)
x_pred_2 = select_2.transform(x_pred)
x_pred_3 = select_3.transform(x_pred)


search_xgb_0.fit(select_x_train_0, y_train_0)
search_xgb_1.fit(select_x_train_1, y_train_1)
search_xgb_2.fit(select_x_train_2, y_train_2)
search_xgb_3.fit(select_x_train_3, y_train_3)


# mae / mse 에 상요하기 위한 y_pred
y_pred_0 = search_xgb_0.predict(select_x_test_0)
y_pred_1 = search_xgb_1.predict(select_x_test_1)
y_pred_2 = search_xgb_2.predict(select_x_test_2)
y_pred_3 = search_xgb_3.predict(select_x_test_3)

r2_0 = search_xgb_0.score(select_x_test_0, y_test_0)
r2_1 = search_xgb_1.score(select_x_test_1, y_test_1)
r2_2 = search_xgb_2.score(select_x_test_2, y_test_2)
r2_3 = search_xgb_3.score(select_x_test_3, y_test_3)


mse_0 = mse(y_test_0, y_pred_0)
mse_1 = mse(y_test_1, y_pred_1)
mse_2 = mse(y_test_2, y_pred_2)
mse_3 = mse(y_test_3, y_pred_3)

mae_0 = mae(y_test_0, y_pred_0)
mae_1 = mae(y_test_1, y_pred_1)
mae_2 = mae(y_test_2, y_pred_2)
mae_3 = mae(y_test_3, y_pred_3)

r2_result = (r2_0 + r2_1 + r2_2 + r2_3)/4
mse_result = (mse_0 + mse_1 + mse_2 + mse_3)/4
mae_result = (mae_0 + mae_1 + mae_2 + mae_3)/4

print("r2_0 :", r2_0)
print("r2_1 :", r2_1)
print("r2_2 :", r2_2)
print("r2_3 :", r2_3)
print()
print("r2_result :", r2_result)
print("===========================================================================================")

print("mse_0 :", mse_0)
print("mse_1 :", mse_1)
print("mse_2 :", mse_2)
print("mse_3 :", mse_3)
print()
print("mse_result :", mse_result)
print("===========================================================================================")

print("mae_0 :", mae_0)
print("mae_1 :", mae_1)
print("mae_2 :", mae_2)
print("mae_3 :", mae_3)
print()
print("mae_result :", mae_result)
print("===========================================================================================")

predict_0 = search_xgb_0.predict(x_pred_0)
predict_1 = search_xgb_1.predict(x_pred_1)
predict_2 = search_xgb_2.predict(x_pred_2)
predict_3 = search_xgb_3.predict(x_pred_3)

predict = [predict_0, predict_1, predict_2, predict_3]
predict = np.array(predict)
print("predict.shape(pre-transpose) :", predict.shape) # (4, 700)
predict = predict.transpose()
print("predict.shape :", predict.shape)


submission = pd.DataFrame(predict, np.arange(2800, 3500))
submission.to_csv(f"./dacon/comp3/submission/0630/submission_h_xgb_{mae_result}.csv", header = ["X", "Y", "M", "V"], index = True, index_label = "id")

